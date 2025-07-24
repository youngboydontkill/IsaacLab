import torch
import math
import time


@torch.jit.script
def joint_to_motor_position(q):
    z_pitch = -0.346
    x_lleq, y_lleq, z_lleq = [0.0385, 0.0283, -0.01]
    x_lreq, y_lreq, z_lreq = [0.0385, -0.0283, -0.01]
    z_llbar = -0.153
    z_lrbar = -0.09
    x_lltd, y_lltd, z_lltd = [0.044, 0.0282564, -0.0100784]
    x_lrtd, y_lrtd, z_lrtd = [0.044, -0.0282633, -0.0100591]
    l_llbar = 0.03
    l_lrbar = 0.03
    l_lltd = 0.193
    l_lrtd = 0.256

    x_rleq, y_rleq, z_rleq = [0.0385, 0.0283, -0.01]
    x_rreq, y_rreq, z_rreq = [0.0385, -0.0283, -0.01]
    z_rlbar = -0.09
    z_rrbar = -0.153
    x_rltd, y_rltd, z_rltd = [0.044, 0.0282633, -0.0100591]
    x_rrtd, y_rrtd, z_rrtd = [0.044, -0.0282564, -0.0100784]
    l_rlbar = 0.03
    l_rrbar = 0.03
    l_rltd = 0.256
    l_rrtd = 0.193

    z_BarKnee = -0.166
    l_tendon = 0.16599763846808185
    l_BarTd = 0.03500002038542263
    l_KneeEq = 0.03500285274088385
    qO_knee = 0.26181698840065826
    qO_bar = 0.3986370610067174

    device = q.device
    q4, q5, q6, q10, q11, q12 = q[:, 3], q[:, 4], q[:, 5], q[:, 9], q[:, 10], q[:, 11]

    sq4 = torch.sin(q4 + qO_knee)
    cq4 = torch.cos(q4 + qO_knee)
    x_BarEq_W = l_KneeEq * sq4
    z_BarEq_W = z_BarKnee + l_KneeEq * cq4
    l_BarEq = torch.sqrt(x_BarEq_W ** 2 + z_BarEq_W ** 2)
    p4 = torch.pi - torch.arccos(
        (l_BarTd ** 2 + l_BarEq ** 2 - l_tendon ** 2) / (2 * l_BarTd * l_BarEq)) + torch.arctan2(x_BarEq_W,
                                                                                                 -z_BarEq_W) - qO_bar

    cq5 = torch.cos(q5)
    sq5 = torch.sin(q5)
    cq6 = torch.cos(q6)
    sq6 = torch.sin(q6)

    # left
    y_LlbarEq_W = y_lleq * cq6 - z_lleq * sq6
    z_LlbarEq_W = -x_lleq * sq5 + y_lleq * sq6 * cq5 - z_llbar + z_lleq * cq5 * cq6 + z_pitch
    y_LrbarEq_W = y_lreq * cq6 - z_lreq * sq6
    z_LrbarEq_W = -x_lreq * sq5 + y_lreq * sq6 * cq5 - z_lrbar + z_lreq * cq5 * cq6 + z_pitch
    x_LltdEq_W = x_lleq * cq5 - x_lltd + y_lleq * sq5 * sq6 + z_lleq * sq5 * cq6
    x_LrtdEq_W = x_lreq * cq5 - x_lrtd + y_lreq * sq5 * sq6 + z_lreq * sq5 * cq6
    b_ll = torch.sqrt(y_LlbarEq_W ** 2 + z_LlbarEq_W ** 2)
    a_ll = l_llbar
    c_ll = torch.sqrt(l_lltd ** 2 - x_LltdEq_W ** 2)
    p6 = torch.arctan2(z_LlbarEq_W, y_LlbarEq_W) + torch.arccos(
        (a_ll ** 2 + b_ll ** 2 - c_ll ** 2) / (2 * a_ll * b_ll)) - math.atan2(z_lltd, y_lltd)
    b_lr = torch.sqrt(y_LrbarEq_W ** 2 + z_LrbarEq_W ** 2)
    a_lr = l_lrbar
    c_lr = torch.sqrt(l_lrtd ** 2 - x_LrtdEq_W ** 2)
    p5 = torch.arctan2(z_LrbarEq_W, y_LrbarEq_W) - torch.arccos(
        (a_lr ** 2 + b_lr ** 2 - c_lr ** 2) / (2 * a_lr * b_lr)) - math.atan2(z_lrtd, y_lrtd)

    # right knee
    sq10 = torch.sin(q10 + qO_knee)
    cq10 = torch.cos(q10 + qO_knee)
    x_BarEq_W = l_KneeEq * sq10
    z_BarEq_W = z_BarKnee + l_KneeEq * cq10
    l_BarEq = torch.sqrt(x_BarEq_W ** 2 + z_BarEq_W ** 2)
    p10 = torch.pi - torch.arccos((l_BarTd ** 2 + l_BarEq ** 2 - l_tendon ** 2) / (2 * l_BarTd * l_BarEq)) + torch.arctan2(
        x_BarEq_W, -z_BarEq_W) - qO_bar

    # right
    cq11 = torch.cos(q11)
    sq11 = torch.sin(q11)
    cq12 = torch.cos(q12)
    sq12 = torch.sin(q12)
    y_RlbarEq_W = y_rleq * cq12 - z_rleq * sq12
    z_RlbarEq_W = -x_rleq * sq11 + y_rleq * sq12 * cq11 - z_rlbar + z_rleq * cq11 * cq12 + z_pitch
    y_RrbarEq_W = y_rreq * cq12 - z_rreq * sq12
    z_RrbarEq_W = -x_rreq * sq11 + y_rreq * sq12 * cq11 - z_rrbar + z_rreq * cq11 * cq12 + z_pitch
    x_RltdEq_W = x_rleq * cq11 - x_rltd + y_rleq * sq11 * sq12 + z_rleq * sq11 * cq12
    x_RrtdEq_W = x_rreq * cq11 - x_rrtd + y_rreq * sq11 * sq12 + z_rreq * sq11 * cq12
    b_rl = torch.sqrt(y_RlbarEq_W ** 2 + z_RlbarEq_W ** 2)
    a_rl = l_rlbar
    c_rl = torch.sqrt(l_rltd ** 2 - x_RltdEq_W ** 2)
    p11 = torch.arctan2(z_RlbarEq_W, y_RlbarEq_W) + torch.arccos(
        (a_rl ** 2 + b_rl ** 2 - c_rl ** 2) / (2 * a_rl * b_rl)) - math.atan2(z_rltd, y_rltd)
    b_rr = torch.sqrt(y_RrbarEq_W ** 2 + z_RrbarEq_W ** 2)
    a_rr = l_rrbar
    c_rr = torch.sqrt(l_rrtd ** 2 - x_RrtdEq_W ** 2)
    p12 = torch.arctan2(z_RrbarEq_W, y_RrbarEq_W) - torch.arccos(
        (a_rr ** 2 + b_rr ** 2 - c_rr ** 2) / (2 * a_rr * b_rr)) - math.atan2(z_rrtd, y_rrtd)

    res = q.clone()
    res[:, 3] = p4
    res[:, 4] = p5
    res[:, 5] = p6
    res[:, 9] = p10
    res[:, 10] = p11
    res[:, 11] = p12
    return res


@torch.jit.script
def get_left_ankle_matrix(q5, q6, p5, p6):
    z_pitch = -0.346
    x_lleq, y_lleq, z_lleq = [0.0385, 0.0283, -0.01]
    x_lreq, y_lreq, z_lreq = [0.0385, -0.0283, -0.01]
    z_llbar = -0.153
    z_lrbar = -0.09
    x_lltd, y_lltd, z_lltd = [0.044, 0.0282564, -0.0100784]
    x_lrtd, y_lrtd, z_lrtd = [0.044, -0.0282633, -0.0100591]

    device = q5.device
    num_envs = q5.shape[0]

    cq5 = torch.cos(q5)
    sq5 = torch.sin(q5)
    cq6 = torch.cos(q6)
    sq6 = torch.sin(q6)
    cp5 = torch.cos(p5)
    sp5 = torch.sin(p5)
    cp6 = torch.cos(p6)
    sp6 = torch.sin(p6)

    JAk_p_WEq = torch.zeros(num_envs, 6, 2, device=device)
    JAk_p_WEq[:, 0, 0] = -x_lleq * sq5 + y_lleq * sq6 * cq5 + z_lleq * cq5 * cq6
    JAk_p_WEq[:, 0, 1] = y_lleq * sq5 * cq6 - z_lleq * sq5 * sq6
    JAk_p_WEq[:, 1, 1] = -y_lleq * sq6 - z_lleq * cq6
    JAk_p_WEq[:, 2, 0] = -x_lleq * cq5 - y_lleq * sq5 * sq6 - z_lleq * sq5 * cq6
    JAk_p_WEq[:, 2, 1] = y_lleq * cq5 * cq6 - z_lleq * sq6 * cq5
    JAk_p_WEq[:, 3, 0] = -x_lreq * sq5 + y_lreq * sq6 * cq5 + z_lreq * cq5 * cq6
    JAk_p_WEq[:, 3, 1] = y_lreq * sq5 * cq6 - z_lreq * sq5 * sq6
    JAk_p_WEq[:, 4, 1] = -y_lreq * sq6 - z_lreq * cq6
    JAk_p_WEq[:, 5, 0] = -x_lreq * cq5 - y_lreq * sq5 * sq6 - z_lreq * sq5 * cq6
    JAk_p_WEq[:, 5, 1] = y_lreq * cq5 * cq6 - z_lreq * sq6 * cq5

    JAct_p_WTd = torch.zeros(num_envs, 6, 2, device=device)
    JAct_p_WTd[:, 1, 0] = -y_lltd * sp6 - z_lltd * cp6
    JAct_p_WTd[:, 2, 0] = y_lltd * cp6 - z_lltd * sp6
    JAct_p_WTd[:, 4, 1] = -y_lrtd * sp5 - z_lrtd * cp5
    JAct_p_WTd[:, 5, 1] = y_lrtd * cp5 - z_lrtd * sp5

    Jxx_l_Td = torch.zeros(num_envs, 2, 6, device=device)
    Jxx_l_Td[:, 0, 0] = x_lleq * cq5 - x_lltd + y_lleq * sq5 * sq6 + z_lleq * sq5 * cq6
    Jxx_l_Td[:, 0, 1] = y_lleq * cq6 - y_lltd * cp6 - z_lleq * sq6 + z_lltd * sp6
    Jxx_l_Td[:, 0,
             2] = -x_lleq * sq5 + y_lleq * sq6 * cq5 - y_lltd * sp6 - z_llbar + z_lleq * cq5 * cq6 - z_lltd * cp6 + z_pitch
    Jxx_l_Td[:, 1, 3] = x_lreq * cq5 - x_lrtd + y_lreq * sq5 * sq6 + z_lreq * sq5 * cq6
    Jxx_l_Td[:, 1, 4] = y_lreq * cq6 - y_lrtd * cp5 - z_lreq * sq6 + z_lrtd * sp5
    Jxx_l_Td[:, 1,
             5] = -x_lreq * sq5 + y_lreq * sq6 * cq5 - y_lrtd * sp5 - z_lrbar + z_lreq * cq5 * cq6 - z_lrtd * cp5 + z_pitch
    return JAk_p_WEq, JAct_p_WTd, Jxx_l_Td


@torch.jit.script
def get_knee_matrix(q4, p4):
    z_BarKnee = -0.166
    l_BarTd = 0.03500002038542263
    l_KneeEq = 0.03500285274088385
    qO_knee = 0.26181698840065826
    qO_bar = 0.3986370610067174

    device = q4.device
    num_envs = q4.shape[0]

    sq4 = torch.sin(q4 + qO_knee)
    cq4 = torch.cos(q4 + qO_knee)
    x_BarEq_W = l_KneeEq * sq4
    z_BarEq_W = z_BarKnee + l_KneeEq * cq4
    sp4 = torch.sin(-p4 - qO_bar)
    cp4 = torch.cos(-p4 - qO_bar)
    x_BarTd_W = l_BarTd * sp4
    z_BarTd_W = l_BarTd * cp4

    Jbar_p_WTd = torch.zeros(num_envs, 2, 1, device=device)
    Jbar_p_WTd[:, 0, 0] = -l_BarTd * cp4
    Jbar_p_WTd[:, 1, 0] = l_BarTd * sp4

    Jknee_p_WEq = torch.zeros(num_envs, 2, 1, device=device)
    Jknee_p_WEq[:, 0, 0] = l_KneeEq * cq4
    Jknee_p_WEq[:, 1, 0] = -l_KneeEq * sq4

    Jxx_l_Td = torch.zeros(num_envs, 1, 2, device=device)
    Jxx_l_Td[:, 0, 0] = x_BarEq_W - x_BarTd_W
    Jxx_l_Td[:, 0, 1] = z_BarEq_W - z_BarTd_W

    return Jknee_p_WEq, Jbar_p_WTd, Jxx_l_Td


@torch.jit.script
def get_right_ankle_matrix(q11, q12, p11, p12):
    z_pitch = -0.346
    x_rleq, y_rleq, z_rleq = [0.0385, 0.0283, -0.01]
    x_rreq, y_rreq, z_rreq = [0.0385, -0.0283, -0.01]
    z_rlbar = -0.09
    z_rrbar = -0.153
    x_rltd, y_rltd, z_rltd = [0.044, 0.0282633, -0.0100591]
    x_rrtd, y_rrtd, z_rrtd = [0.044, -0.0282564, -0.0100784]

    device = q11.device
    num_envs = q11.shape[0]

    cq11 = torch.cos(q11)
    sq11 = torch.sin(q11)
    cq12 = torch.cos(q12)
    sq12 = torch.sin(q12)
    cp11 = torch.cos(p11)
    sp11 = torch.sin(p11)
    cp12 = torch.cos(p12)
    sp12 = torch.sin(p12)

    JAk_p_WEq = torch.zeros(num_envs, 6, 2, device=device)
    JAk_p_WEq[:, 0, 0] = -x_rleq * sq11 + y_rleq * sq12 * cq11 + z_rleq * cq11 * cq12
    JAk_p_WEq[:, 0, 1] = y_rleq * sq11 * cq12 - z_rleq * sq11 * sq12
    JAk_p_WEq[:, 1, 1] = -y_rleq * sq12 - z_rleq * cq12
    JAk_p_WEq[:, 2, 0] = -x_rleq * cq11 - y_rleq * sq11 * sq12 - z_rleq * sq11 * cq12
    JAk_p_WEq[:, 2, 1] = y_rleq * cq11 * cq12 - z_rleq * sq12 * cq11
    JAk_p_WEq[:, 3, 0] = -x_rreq * sq11 + y_rreq * sq12 * cq11 + z_rreq * cq11 * cq12
    JAk_p_WEq[:, 3, 1] = y_rreq * sq11 * cq12 - z_rreq * sq11 * sq12
    JAk_p_WEq[:, 4, 1] = -y_rreq * sq12 - z_rreq * cq12
    JAk_p_WEq[:, 5, 0] = -x_rreq * cq11 - y_rreq * sq11 * sq12 - z_rreq * sq11 * cq12
    JAk_p_WEq[:, 5, 1] = y_rreq * cq11 * cq12 - z_rreq * sq12 * cq11

    JAct_p_WTd = torch.zeros(num_envs, 6, 2, device=device)
    JAct_p_WTd[:, 1, 0] = -y_rltd * sp11 - z_rltd * cp11
    JAct_p_WTd[:, 2, 0] = y_rltd * cp11 - z_rltd * sp11
    JAct_p_WTd[:, 4, 1] = -y_rrtd * sp12 - z_rrtd * cp12
    JAct_p_WTd[:, 5, 1] = y_rrtd * cp12 - z_rrtd * sp12

    Jxx_l_Td = torch.zeros(num_envs, 2, 6, device=device)
    Jxx_l_Td[:, 0, 0] = x_rleq * cq11 - x_rltd + y_rleq * sq11 * sq12 + z_rleq * sq11 * cq12
    Jxx_l_Td[:, 0, 1] = y_rleq * cq12 - y_rltd * cp11 - z_rleq * sq12 + z_rltd * sp11
    Jxx_l_Td[:, 0,
             2] = -x_rleq * sq11 + y_rleq * sq12 * cq11 - y_rltd * sp11 - z_rlbar + z_rleq * cq11 * cq12 - z_rltd * cp11 + z_pitch
    Jxx_l_Td[:, 1, 3] = x_rreq * cq11 - x_rrtd + y_rreq * sq11 * sq12 + z_rreq * sq11 * cq12
    Jxx_l_Td[:, 1, 4] = y_rreq * cq12 - y_rrtd * cp12 - z_rreq * sq12 + z_rrtd * sp12
    Jxx_l_Td[:, 1,
             5] = -x_rreq * sq11 + y_rreq * sq12 * cq11 - y_rrtd * sp12 - z_rrbar + z_rreq * cq11 * cq12 - z_rrtd * cp12 + z_pitch

    return JAk_p_WEq, JAct_p_WTd, Jxx_l_Td


@torch.jit.script
def joint_to_motor_velocity(q, p, dq):
    num_envs = q.shape[0]
    q4, q5, q6, q10, q11, q12 = q[:, 3], q[:, 4], q[:, 5], q[:, 9], q[:, 10], q[:, 11]
    p4, p5, p6, p10, p11, p12 = p[:, 3], p[:, 4], p[:, 5], p[:, 9], p[:, 10], p[:, 11]
    dq4, dq5, dq6, dq10, dq11, dq12 = dq[:, 3], dq[:, 4], dq[:, 5], dq[:, 9], dq[:, 10], dq[:, 11]
    res = dq.clone()

    # left
    Jknee_p_WEq, Jbar_p_WTd, Jxx_l_Td = get_knee_matrix(q4, p4)
    Jknee_bar = 1. / (Jxx_l_Td @ Jbar_p_WTd) * (Jxx_l_Td @ Jknee_p_WEq)
    res[:, 3] = Jknee_bar.squeeze(-1).squeeze(-1) * dq4

    JAk_p_WEq, JAct_p_WTd, Jxx_l_Td = get_left_ankle_matrix(q5, q6, p5, p6)
    AkDt = torch.stack([dq5, dq6], dim=1)
    JAk_Act = torch.linalg.inv(Jxx_l_Td @ JAct_p_WTd) @ (Jxx_l_Td @ JAk_p_WEq)
    ActDt = (JAk_Act @ AkDt.unsqueeze(-1)).squeeze(-1)
    res[:, 4:6] = ActDt

    # right
    Jknee_p_WEq, Jbar_p_WTd, Jxx_l_Td = get_knee_matrix(q10, p10)
    Jknee_bar = 1. / (Jxx_l_Td @ Jbar_p_WTd) * (Jxx_l_Td @ Jknee_p_WEq)
    res[:, 9] = Jknee_bar.squeeze(-1).squeeze(-1) * dq10

    JAk_p_WEq, JAct_p_WTd, Jxx_l_Td = get_right_ankle_matrix(q11, q12, p11, p12)
    AkDt = torch.stack([dq11, dq12], dim=1)
    JAk_Act = torch.linalg.inv(Jxx_l_Td @ JAct_p_WTd) @ (Jxx_l_Td @ JAk_p_WEq)
    ActDt = (JAk_Act @ AkDt.unsqueeze(-1)).squeeze(-1)
    res[:, 10:12] = ActDt
    return res


@torch.jit.script
def motor_to_joint_torque(q, p, i):
    num_envs = q.shape[0]
    q4, q5, q6, q10, q11, q12 = q[:, 3], q[:, 4], q[:, 5], q[:, 9], q[:, 10], q[:, 11]
    p4, p5, p6, p10, p11, p12 = p[:, 3], p[:, 4], p[:, 5], p[:, 9], p[:, 10], p[:, 11]
    i4, i5, i6, i10, i11, i12 = i[:, 3], i[:, 4], i[:, 5], i[:, 9], i[:, 10], i[:, 11]

    res = i.clone()

    # left
    Jknee_p_WEq, Jbar_p_WTd, Jxx_l_Td = get_knee_matrix(q4, p4)
    Jknee_bar = 1. / (Jxx_l_Td @ Jbar_p_WTd) * (Jxx_l_Td @ Jknee_p_WEq)
    res[:, 3] = Jknee_bar.squeeze(-1).squeeze(-1) * i4

    JAk_p_WEq, JAct_p_WTd, Jxx_l_Td = get_left_ankle_matrix(q5, q6, p5, p6)
    JAk_Act = torch.linalg.inv(Jxx_l_Td @ JAct_p_WTd) @ (Jxx_l_Td @ JAk_p_WEq)
    tau_Act = torch.stack([i5, i6], dim=1)
    tau_Ak = JAk_Act.transpose(1, 2) @ tau_Act.unsqueeze(-1)
    res[:, 4:6] = tau_Ak.squeeze(-1)

    # right
    Jknee_p_WEq, Jbar_p_WTd, Jxx_l_Td = get_knee_matrix(q10, p10)
    Jknee_bar = 1. / (Jxx_l_Td @ Jbar_p_WTd) * (Jxx_l_Td @ Jknee_p_WEq)
    res[:, 9] = Jknee_bar.squeeze(-1).squeeze(-1) * i10

    JAk_p_WEq, JAct_p_WTd, Jxx_l_Td = get_right_ankle_matrix(q11, q12, p11, p12)
    JAk_Act = torch.linalg.inv(Jxx_l_Td @ JAct_p_WTd) @ (Jxx_l_Td @ JAk_p_WEq)
    tau_Act = torch.stack([i11, i12], dim=1)
    tau_Ak = JAk_Act.transpose(1, 2) @ tau_Act.unsqueeze(-1)
    res[:, 10:12] = tau_Ak.squeeze(-1)
    return res


@torch.jit.script
def get_joint_dumping_torque(q, p, kd, qd):
    num_envs = q.shape[0]
    q4, q5, q6, q10, q11, q12 = q[:, 3], q[:, 4], q[:, 5], q[:, 9], q[:, 10], q[:, 11]
    p4, p5, p6, p10, p11, p12 = p[:, 3], p[:, 4], p[:, 5], p[:, 9], p[:, 10], p[:, 11]
    kd4, kd5, kd6, kd10, kd11, kd12 = kd[:, 3], kd[:, 4], kd[:, 5], kd[:, 9], kd[:, 10], kd[:, 11]

    res = qd * kd

    # left
    Jknee_p_WEq, Jbar_p_WTd, Jxx_l_Td = get_knee_matrix(q4, p4)
    Jknee_bar = 1. / (Jxx_l_Td @ Jbar_p_WTd) * (Jxx_l_Td @ Jknee_p_WEq)
    res[:, 3] = Jknee_bar.squeeze(-1).squeeze(-1) ** 2 * kd4 * qd[:, 3]

    JAk_p_WEq, JAct_p_WTd, Jxx_l_Td = get_left_ankle_matrix(q5, q6, p5, p6)
    JAk_Act = torch.linalg.inv(Jxx_l_Td @ JAct_p_WTd) @ (Jxx_l_Td @ JAk_p_WEq)
    kd_Act = torch.zeros(num_envs, 2, 2).to(q.device)
    kd_Act[:, 0, 0] = kd5
    kd_Act[:, 1, 1] = kd6
    torque = JAk_Act.transpose(1, 2) @ kd_Act @ JAk_Act @ qd[:, 4:6].unsqueeze(-1)
    res[:, 4:6] = torque.squeeze(-1)

    # right
    Jknee_p_WEq, Jbar_p_WTd, Jxx_l_Td = get_knee_matrix(q10, p10)
    Jknee_bar = 1. / (Jxx_l_Td @ Jbar_p_WTd) * (Jxx_l_Td @ Jknee_p_WEq)
    res[:, 9] = Jknee_bar.squeeze(-1).squeeze(-1) ** 2 * kd10 * qd[:, 9]

    JAk_p_WEq, JAct_p_WTd, Jxx_l_Td = get_right_ankle_matrix(q11, q12, p11, p12)
    JAk_Act = torch.linalg.inv(Jxx_l_Td @ JAct_p_WTd) @ (Jxx_l_Td @ JAk_p_WEq)
    kd_Act = torch.zeros(num_envs, 2, 2).to(q.device)
    kd_Act[:, 0, 0] = kd11
    kd_Act[:, 1, 1] = kd12
    torque = JAk_Act.transpose(1, 2) @ kd_Act @ JAk_Act @ qd[:, 10:12].unsqueeze(-1)
    res[:, 10:12] = torque.squeeze(-1)
    return res


@torch.jit.script
def is_ankle_pos_legal(points):
    device = points.device
    vertices = torch.tensor([
        [0.34, 0.],
        [-0.26, 0.8],
        [-0.87, 0.],
        [-0.26, -0.8],
    ], device=device)
    distances = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        edge_vector = p2 - p1
        point_vector = points - p1
        distance = edge_vector[0] * point_vector[:, 1] - edge_vector[1] * point_vector[:, 0]
        distances.append(distance)
    distances = torch.stack(distances, dim=1)

    is_in_diamond = torch.logical_or(
        torch.all(distances >= 0, dim=1),
        torch.all(distances <= 0, dim=1)
    )
    return is_in_diamond


def generate_diamond_points(num_points=8092, ratio=10):
    vertices = torch.tensor([
        [0.34, 0.],
        [-0.26, 0.8],
        [-0.87, 0.],
        [-0.26, -0.8],
    ])
    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
    points = torch.rand(num_points * ratio, 2) * torch.tensor([max_x - min_x, max_y - min_y]) + torch.tensor([min_x, min_y])
    if_in = is_ankle_pos_legal(points)
    assert torch.sum(if_in) >= num_points
    return points[if_in][:num_points]


if __name__ == '__main__':
    # seed
    torch.manual_seed(0)

    legal_ankle_pos = generate_diamond_points()

    joint_pos = torch.rand(8092, 12).to("cuda")
    joint_pos[:, 4:6] = legal_ankle_pos.clone()
    joint_pos[:, 10:12] = legal_ankle_pos.clone()
    joint_vel = torch.rand(8092, 12).to("cuda")
    # motor_torque = torch.rand(8092, 12).to("cuda")
    motor_kd = torch.ones(8092, 12).to("cuda")

    # print(joint_pos)
    motor_pos = joint_to_motor_position(joint_pos)
    # print(motor_pos)

    # print(joint_vel)
    motor_vel = joint_to_motor_velocity(joint_pos, motor_pos, joint_vel)
    # print(motor_vel)
    #
    motor_torque = motor_kd * motor_vel

    # print(motor_torque)
    joint_torque = motor_to_joint_torque(joint_pos, motor_pos, motor_torque)
    # print(joint_torque)

    joint_torque2 = get_joint_dumping_torque(joint_pos, motor_pos, motor_kd, joint_vel)
    assert torch.allclose(joint_torque, joint_torque2, atol=0.01)
