import torch 
from tensordict import TensorDict 
import numpy as np 

def generate_height_scan_mirror(start_idx=140, rows=17, cols=11):
    mirror_indices = []
    for row in range(rows):
        for col in range(cols):
            mirror_col = cols - 1 - col
            mirror_idx = start_idx + row * cols + mirror_col
            mirror_indices.append(mirror_idx)
    mirror_signs = [1] * (rows * cols)
    return mirror_indices, mirror_signs

def mirror_scan_height(scan_height: torch.Tensor):
    assert(scan_height.shape[-1] == 3)
    L = scan_height.shape[-3]
    W = scan_height.shape[-2]
    mirror_scan = torch.zeros_like(scan_height)
    mirror_scan[...,0] = scan_height[...,0]
    mirror_scan[...,1] = -scan_height[...,1]
    mirror_scan[...,2] = scan_height[...,2].flip(-1)
    return mirror_scan

class SymmetryAug2D:
    """
    symmetry augmentation for 2D Tensor, which shape is [B,H*d]
    """
    OBS_MIRROR_REGIST_KEYS = {}  # 注册的key
    OBS_MIRROR_INDICES_DICT = {}  # 注册的key对应的镜像索引    
    OBS_MIRROR_SIGNS_DICT = {}  # 注册的key对应的
    # for lab sytle joint indices for kuavo s42
    ACT_MIRROR_INDICES = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19]
    ACT_MIRROR_SIGNS = [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1]
    # supprot keys 
    OBS_MIRROR_SUPPORT_DICT = {
            "base_ang_vel":[[0, 1, 2],[-1,1,-1]],
            "gravity":[[0,1,2],[1,-1,1]],
            "cmd":[[0,1,2,3],[1, -1, -1, 1]],
            "cmd_vel":[[0,1,2],[1, -1, -1]],
            "hugwbc_cmd":[[0],[1]],
            "joint_pos":[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19],
                        [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1]],
            "joint_vel":[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19],
                        [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1]],
            "action":[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19],
                        [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1]],
            "base_lin_vel": [[0, 1, 2],[1, -1, 1]],
            # 高程图，1.6m x 1.0m，分辨率0.1m，共17x11个点，排列顺序为xy，所以关于y对称就是每隔17个点为一列，把这11列倒序排列即可。符号不变。
            "height_scan": generate_height_scan_mirror(0,17,11),
            "joint_torques": [[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19],
                        [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1]],
            "joint_accs": [[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19],
                        [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1]],
            "feet_lin_vel": [[3, 4, 5, 0, 1, 2],[1, -1, 1, 1, -1, 1]],
            "feet_contact_force": [[3, 4, 5, 0, 1, 2],[1, -1, 1, 1, -1, 1]],
            "base_mass_rel": [[0,],[1,]],
            "rigid_body_material": [[3, 4, 5, 0, 1, 2],[1, 1, 1, 1, 1, 1]],
            "base_com": [[0,1,2],[1,-1,1]],
            "action_delay": [[0,],[1,]],
            "push_force": [[0,1,2],[1,-1,1]],
            "push_torque": [[0, 1, 2],[-1,1,-1]],
            "feet_heights": [[1,0],[1,1]],
            "feet_air_times": [[1,0],[1,1]],
    }
    
    @staticmethod
    def mirror_tensors(tensor, mirror_indices, mirror_signs):
        mirrored_tensor = tensor[..., mirror_indices]
        mirror_signs_tensor = torch.tensor(mirror_signs, dtype=tensor.dtype, device=tensor.device)
        mirrored_tensor = mirrored_tensor * mirror_signs_tensor
        return mirrored_tensor
    
    @classmethod
    def clear(cls):
        cls.OBS_MIRROR_REGIST_KEYS = {}
        cls.OBS_MIRROR_INDICES_DICT = {}
        cls.OBS_MIRROR_SIGNS_DICT = {}

    @classmethod
    def register_obs(cls, key:str, obsgroup:list,history_len:int=1):
        # step 1 : check is supported
        # assert key in cls.OBS_MIRROR_SUPPORT_DICT.keys(), f"key {key} is not supported"
        # step 2 : register
        if key not in cls.OBS_MIRROR_REGIST_KEYS.keys():
            cls.OBS_MIRROR_REGIST_KEYS[key] = obsgroup
        else:
            for obsterm in obsgroup:
                assert obsterm not in cls.OBS_MIRROR_REGIST_KEYS[key], f"Obsterm {obsterm} is already registered"
            cls.OBS_MIRROR_REGIST_KEYS[key].extend(obsgroup)
        # step 3 : generate mirror indices and signs 
        idx = 0
        obs_mirror_indices = []
        obs_mirror_signs = []
        for obsterm in cls.OBS_MIRROR_REGIST_KEYS[key]:
            n = len(cls.OBS_MIRROR_SUPPORT_DICT[obsterm][0])
            for i in range(history_len):
                start_idx = idx 
                obs_mirror_indices.extend([i + start_idx for i in cls.OBS_MIRROR_SUPPORT_DICT[obsterm][0]])
                obs_mirror_signs.extend(cls.OBS_MIRROR_SUPPORT_DICT[obsterm][1])
                idx += n
        cls.OBS_MIRROR_INDICES_DICT[key] = obs_mirror_indices
        cls.OBS_MIRROR_SIGNS_DICT[key] = obs_mirror_signs

    @classmethod
    def register_policy_obs(cls,obsgroup:list,history_len=1):        
        """
        :brief : this method is used for register obs group for policy, used for rsl rl <= 3.0.1 version
        """
        k = 'policy'
        cls.register_obs(k,obsgroup,history_len)
    
    @classmethod
    def register_critic_obs(cls,obsgroup:list,history_len=1):        
        """
        :brief : this method is used for register obs group for critic, used for rsl rl <= 3.0.1 version
        """
        k = 'critic'
        cls.register_obs(k,obsgroup,history_len)
    
    @classmethod
    def data_augmentation_tensor(cls,env,obs:torch.Tensor,actions:torch.Tensor,obs_type:str)->tuple:
        """
        :brief : this method is used for data augmentation with rsl rl version <= 3.0.1, which VecEnv.get_observation() 
        returns (torch.Tensor, dict) instead of TensorDict.
        """
        if obs is None:
            obs_aug = None
        else:
            if obs_type == 'policy':
                policy_mirror_indices = cls.OBS_MIRROR_INDICES_DICT['policy']
                policy_mirror_signs = cls.OBS_MIRROR_SIGNS_DICT['policy']
                obs_aug = torch.cat((obs, cls.mirror_tensors(obs, policy_mirror_indices, policy_mirror_signs)), dim=0)
            elif obs_type == 'critic':
                critic_mirror_indices = cls.OBS_MIRROR_INDICES_DICT['critic']
                critic_mirror_signs = cls.OBS_MIRROR_SIGNS_DICT['critic']
                obs_aug = torch.cat((obs, cls.mirror_tensors(obs, critic_mirror_indices, critic_mirror_signs)), dim=0)
            else:
                raise ValueError(f"Mirror logic for observation type '{obs_type}' not implemented")
        if actions is None:
            actions_aug = None
        else:
            actions_aug = torch.cat((actions, cls.mirror_tensors(actions, cls.ACT_MIRROR_INDICES, cls.ACT_MIRROR_SIGNS)), dim=0)
        return obs_aug, actions_aug

    @classmethod
    def data_augmentation_dict(cls,env, obs:TensorDict, actions:torch.Tensor)->tuple:
        """
        :brief : this method is used for data augmentation with rsl rl version > 3.0.1, which VecEnv.get_observation() 
        returns a TensorDict
        """
        if obs is None:
            obs_aug = None 
        else:
            B = obs.batch_size[0]  # tensordict 特有的
            new_batch_size = [2*B] # 假设你在 dim=1 拼接
            obs_aug = TensorDict({}, batch_size=new_batch_size, device=obs.device)
            for k in obs.keys():
                assert k in cls.OBS_MIRROR_REGIST_KEYS.keys(), f"Observation key '{k}' not in mirror regist keys"
                mirror_indices = cls.OBS_MIRROR_INDICES_DICT[k]
                mirror_signs = cls.OBS_MIRROR_SIGNS_DICT[k]
                obs_k_aug = torch.cat((obs[k], cls.mirror_tensors(obs[k], 
                    mirror_indices, mirror_signs)), dim=0)
                obs_aug[k] = obs_k_aug
        if actions is None:
            actions_aug = None
        else:
            actions_aug = torch.cat((actions, cls.mirror_tensors(actions, cls.ACT_MIRROR_INDICES, cls.ACT_MIRROR_SIGNS)), dim=0)
        return obs_aug, actions_aug

class SymmetryAug(SymmetryAug2D):
    """
    this class is used for data augmentation without flatten_history, which obs tensor's shape is 
    [B,H,D,...]. Especially for image, which shape is [B,H,L,W,C](we use channel last formation), we 
    only need to mirror the image along the width axis.
    """
    # IMAGE'S KEYS , shape = [B,H,L,W,C] , L和机器人的x轴重合,只对W进行flip. 特别的,针对map scan,我们对x不变,y负号,z进行flip
    OBS_HIGH_DIM_AUG_DICT = {
        # "perception" : mirror_scan_height
    }

    @classmethod
    def register_obs(cls, key:str, obsgroup:list,history_len:int=1):
        """
        :param key: the key of observation
        :param obsgroup: the group of observation
        :param history_len: the length of history, for SymmetryAug, history_len is not used
        """
        # step 1 : check is supported
        # 针对高维输入需要用自己定义的aug函数
        assert key != 'perception', f"High Dimensional input use register_high_dim_obs!"
        # assert key in cls.OBS_MIRROR_SUPPORT_DICT.keys(), f"key {key} is not supported"
        # step 2 : register
        if key not in cls.OBS_MIRROR_REGIST_KEYS.keys():
            cls.OBS_MIRROR_REGIST_KEYS[key] = obsgroup
        else:
            for obsterm in obsgroup:
                assert obsterm not in cls.OBS_MIRROR_REGIST_KEYS[key], f"Obsterm {obsterm} is already registered"
            cls.OBS_MIRROR_REGIST_KEYS[key].extend(obsgroup)
        # step 3 : generate mirror indices and signs 
        idx = 0
        obs_mirror_indices = []
        obs_mirror_signs = []
        for obsterm in cls.OBS_MIRROR_REGIST_KEYS[key]:
            n = len(cls.OBS_MIRROR_SUPPORT_DICT[obsterm][0])
            obs_mirror_indices.extend([i + idx for i in cls.OBS_MIRROR_SUPPORT_DICT[obsterm][0]])
            obs_mirror_signs.extend(cls.OBS_MIRROR_SUPPORT_DICT[obsterm][1])
            idx += n
        cls.OBS_MIRROR_INDICES_DICT[key] = obs_mirror_indices
        cls.OBS_MIRROR_SIGNS_DICT[key] = obs_mirror_signs

    @classmethod
    def register_obs_high_dim(cls, key:str, high_dim_obs_aug):
        cls.OBS_MIRROR_REGIST_KEYS[key] = []
        cls.OBS_HIGH_DIM_AUG_DICT[key] = high_dim_obs_aug
    
    @classmethod
    def data_augmentation_dict(cls,env, obs:TensorDict, actions:torch.Tensor)->tuple:
        """
        :brief : this method is used for data augmentation with rsl rl version > 3.0.1, which VecEnv.get_observation() 
        returns a TensorDict
        """
        if obs is None:
            obs_aug = None 
        else:
            # observations = self.observations.flatten(0, 1)->这里接受到的obs的batch size就已经发生变化了
            B = obs.batch_size[0]  # tensordict 特有的
            new_batch_size = [2*B] # 假设你在 dim=1 拼接
            obs_aug = TensorDict({}, batch_size=new_batch_size, device=obs.device)
            for k in obs.keys():
                assert k in cls.OBS_MIRROR_REGIST_KEYS.keys(), f"Observation key '{k}' not in mirror regist keys"
                if (k in cls.OBS_HIGH_DIM_AUG_DICT.keys()):
                    obs_k_aug = cls.OBS_HIGH_DIM_AUG_DICT[k](obs[k])
                    obs_aug[k] = torch.cat((obs[k], obs_k_aug), dim=0)
                else:
                    mirror_indices = cls.OBS_MIRROR_INDICES_DICT[k]
                    mirror_signs = cls.OBS_MIRROR_SIGNS_DICT[k]
                    obs_k_aug = torch.cat((obs[k], cls.mirror_tensors(obs[k], mirror_indices, mirror_signs)), dim=0)
                    obs_aug[k] = obs_k_aug
        if actions is None:
            actions_aug = None
        else:
            actions_aug = torch.cat((actions, cls.mirror_tensors(actions, cls.ACT_MIRROR_INDICES, cls.ACT_MIRROR_SIGNS)), dim=0)
        return obs_aug, actions_aug