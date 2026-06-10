import numpy as np

OBSTERMGYM2LAB = {
            "base_ang_vel":[0,1,2],
            "gravity":[0,1,2],
            "cmd":[0,1,2,3],
            "cmd_vel":[0,1,2],
            "hugwbc_cmd":[0],
            "joint_pos": [i for i in range(27)],
            "joint_vel": [i for i in range(27)],
            "action": [i for i in range(27)],
        }

def generate_lab_obs_indices(obs_keys:dict,history_len:int)->np.array:
    """
    将gym的风格的观测转换为lab的索引 obs[indices]->[obs_1[0:h],obs_2[0:h],...]
    """
    policy_obs_idx = 0 
    single_gym2lab_dict = {}
    single_gym2lab = np.array([])
    # 先构建(history, N)的矩阵，然后将其改为列存储
    for k in obs_keys:
        term_gym2lab = np.array(OBSTERMGYM2LAB[k])
        n = term_gym2lab.shape[0]
        single_gym2lab_dict[k] = [policy_obs_idx, policy_obs_idx+n]
        single_gym2lab = np.concatenate((single_gym2lab, term_gym2lab+policy_obs_idx), axis=-1)
        policy_obs_idx += n
    gym2lab_mat = [single_gym2lab + i * single_gym2lab.shape[0] for i in range(history_len)]
    gym2lab_mat = np.array(gym2lab_mat).reshape(history_len,-1)  # 此时每行是一个
    gym2lab = np.array([])

    for k in obs_keys:
        start_idx = single_gym2lab_dict[k][0]
        end_idx = single_gym2lab_dict[k][1]
        for i in range(history_len):
            obs_i = gym2lab_mat[i]
            gym2lab = np.concatenate((gym2lab, obs_i[start_idx:end_idx]))
    gym2lab = np.array(gym2lab).reshape(-1)
    return gym2lab.astype(np.int32)

POLICY_KEYS = ["base_ang_vel","gravity","cmd","joint_pos","joint_vel","action"]
HISTORY_LEN = 5
PERM = generate_lab_obs_indices(POLICY_KEYS,history_len=HISTORY_LEN)

singleObs = ["omega_x","omega_y","omega_z",
             "gravity_x","gravity_y","gravity_z",
             "cmd_x","cmd_y","cmd_yaw","cmd_stand"] + \
            ["pos_{}".format(i) for i in range(27)] + \
            ["vel_{}".format(i) for i in range(27)] + \
            ["action_{}".format(i) for i in range(27)]

gymObs = []
for i in range(HISTORY_LEN):
    gymObs += [k+"_"+str(i) for k in singleObs]

gymObsArray = np.array(gymObs)
print(gymObsArray[PERM])
#   94  95  96 185 186 187 276 277 278 367 368 369   6   7   8   9  97  98
#   99 100 188 189 190 191 279 280 281 282 370 371 372 373  10  11  12  13
#   14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
#   32  33  34  35  36 101 102 103 104 105 106 107 108 109 110 111 112 113
#  114 115 116 117 118 119 120 121 122 123 124 125 126 127 192 193 194 195
#  196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213
#  214 215 216 217 218 283 284 285 286 287 288 289 290 291 292 293 294 295
#  296 297 298 299 300 301 302 303 304 305 306 307 308 309 374 375 376 377
#  378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395
#  396 397 398 399 400  37  38  39  40  41  42  43  44  45  46  47  48  49
#   50  51  52  53  54  55  56  57  58  59  60  61  62  63 128 129 130 131
#  132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
#  150 151 152 153 154 219 220 221 222 223 224 225 226 227 228 229 230 231
#  232 233 234 235 236 237 238 239 240 241 242 243 244 245 310 311 312 313
#  314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331
#  332 333 334 335 336 401 402 403 404 405 406 407 408 409 410 411 412 413
#  414 415 416 417 418 419 420 421 422 423 424 425 426 427  64  65  66  67
#   68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85
#   86  87  88  89  90 155 156 157 158 159 160 161 162 163 164 165 166 167
#  168 169 170 171 172 173 174 175 176 177 178 179 180 181 246 247 248 249
#  250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267
#  268 269 270 271 272 337 338 339 340 341 342 343 344 345 346 347 348 349
#  350 351 352 353 354 355 356 357 358 359 360 361 362 363 428 429 430 431
#  432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449
#  450 451 452 453 454]
