import os
import csv
import json
import torch
import joblib
import numpy as np
import casadi as ca
import l4casadi as l4c
from sklearn.preprocessing import StandardScaler
from car_control_pkg.car_predictor import CarPredictor
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

def loadConfig():
    with open('/home/liangh/mpc/config.json', 'r') as f:
        config = json.load(f)
    return config

def computeTarget(u_tensor, x_data, A, B, dt=0.1):
    x_dot = torch.bmm(A, x_data) + torch.bmm(B, u_tensor)  # x' = Ax + Bu
    next_x = x_data + x_dot * dt  # Euler integration: x_next = x + x' * dt
    return next_x

def normalize(data, name, scaler):
    if name == "u":
        data = scaler.transform(data)
    elif name == "state":
        data = np.array(data)
        data[:2] = scaler.transform(data[:2].reshape(-1, 2)).reshape(data[:2].shape)
    elif name == "path":
        data = np.array(data)
        data[:, :2] = scaler.transform(data[:, :2].reshape(-1, 2)).reshape(data[:, :2].shape)
    return data

def denormalize(data, name, scaler):
    if name == "u":
        data = scaler.inverse_transform(data.reshape(-1, 4)).reshape(data.shape)
    elif name == "x":
        data = scaler.inverse_transform(data)
    elif name =="state":
        data = np.array(data)
        data[:2] = scaler.inverse_transform(data[:2].reshape(-1, 2)).reshape(data[:2].shape)
    return data

def _wheel_pair_from_control(u_sym):
    left_velocity = 0.5 * (u_sym[0] + u_sym[2])
    right_velocity = 0.5 * (u_sym[1] + u_sym[3])
    return left_velocity, right_velocity

def _build_nn_input(x_sym, u_sym):
    current_left_velocity = x_sym[4]
    current_right_velocity = x_sym[5]
    next_left_velocity, next_right_velocity = _wheel_pair_from_control(u_sym)
    return ca.vertcat(
        current_left_velocity,
        current_right_velocity,
        next_left_velocity,
        next_right_velocity,
    )

def _predict_next_state(x_sym, u_sym, delta_sym):
    heading = ca.atan2(x_sym[2], x_sym[3])
    x_pos_next = x_sym[0] + delta_sym[0]
    z_pos_next = x_sym[1] + delta_sym[1]
    next_heading = heading + delta_sym[2]
    next_heading = ca.atan2(ca.sin(next_heading), ca.cos(next_heading))  # normalize to [-pi, pi]
    next_left_velocity, next_right_velocity = _wheel_pair_from_control(u_sym)

    return ca.vertcat(
        x_pos_next,
        z_pos_next,
        ca.sin(next_heading),
        ca.cos(next_heading),
        next_left_velocity,
        next_right_velocity,
    )

def angleToDegree(data):
    sin_component = data[:, 2, :]
    cos_component = data[:, 3, :]
    angles = np.arctan2(sin_component, cos_component)
    angles_degrees = np.degrees(angles)
    angles_degrees = (angles_degrees + 360) % 360  # Normalize to [0, 360)
    data[:, 2, :] = angles_degrees
    data[:, 3, :] = 0  # set the cosine component to zero
    return data

def loadModelFunc(model_path, dt):
    # OPTIMIZATION: Prevent PyTorch from wasting cycles managing multi-threading
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model = CarPredictor()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model.load_state_dict(torch.load(f'{model_path}/model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    l4c_model = l4c.L4CasADi(model, device=device)
    nn_input_sym = ca.SX.sym('nn_input', 4)
    delta_sym = l4c_model(nn_input_sym.T).T
    nn_model_func = ca.Function('nn_model_func', [nn_input_sym], [delta_sym], ['nn_input'], ['delta_x'])

    return nn_model_func, l4c_model.shared_lib_dir, l4c_model.name

def createMpcSolver(nn_model_func, N=10, dt=0.1):
    opti = ca.Opti()

    u_pred = opti.variable(N, 4)  # Control inputs over the horizon
    next_x_pred = opti.variable(N + 1, 6)  # Predicted states over the horizon

    current_x = opti.parameter(6) # Current state parameter
    current_wheel = opti.parameter(2) # Current wheel velocity parameter
    target_path = opti.parameter(N+1, 2) # Target path parameter

    cost = 0
    control_cost_weight = ca.diag(ca.DM([0.01, 0.01, 0.01, 0.01]))
    for t in range(N+1):
        position_error = next_x_pred[t, :2] - target_path[t, :]
        cost += ca.sumsqr(position_error)
    
    for t in range(N):
        control_effort = u_pred[t, :]
        cost += ca.mtimes([control_effort, control_cost_weight, control_effort.T])

    opti.minimize(cost)

    for t in range(N):
        u_t = u_pred[t, :]
        x_t = next_x_pred[t, :]
        nn_input_t = _build_nn_input(x_t.T, u_t.T)
        delta_t = nn_model_func(nn_input_t)
        x_next_pred_t = _predict_next_state(x_t.T, u_t.T, delta_t)
        opti.subject_to(next_x_pred[t+1, :] == x_next_pred_t.T)
    opti.subject_to(next_x_pred[0, :] == current_x.T)
    opti.subject_to(opti.bounded(-1.0, u_pred, 1.0))
    opts = {"ipopt.print_level":0, "print_time":0}
    opti.solver('ipopt', opts)
    return opti, u_pred, next_x_pred, current_x, current_wheel, target_path

def createAcadosSolver(nn_model_func, lib_dir, lib_name, N, dt):
    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = 'car_model'

    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 4)
    p = ca.SX.sym('p', 2)
    nn_input = _build_nn_input(x, u)
    delta_sym = nn_model_func(nn_input)
    x_next = _predict_next_state(x, u, delta_sym)

    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next
    ocp.model = model

    ocp.solver_options.N_horizon = N
    Tf = N * dt  # (N steps) * (dt s/step)
    ocp.solver_options.tf = Tf

    Q_pos = np.diag([1000.0, 1000.0]) # Position cost
    R_ctrl = np.diag([0.01, 0.01, 0.01, 0.01]) # Control effort cost

    position_error = x[:2] - p[:2]

    stage_cost_expr = ca.mtimes([position_error.T, ca.DM(Q_pos), position_error]) \
                      + ca.mtimes([u.T, ca.DM(R_ctrl), u])

    terminal_cost_expr = ca.mtimes([position_error.T, ca.DM(Q_pos), position_error])

    # --- Set cost for STAGE 0 ---
    ocp.cost.cost_type_0 = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_0 = stage_cost_expr

    # --- Set cost for STAGES 1 to N-1 ---
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = stage_cost_expr

    # --- Set cost for STAGE N (Terminal) ---
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = terminal_cost_expr

    ocp.constraints.x0 = np.zeros(6)
    ocp.parameter_values = np.zeros(2)
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, -1.0])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.model_external_shared_lib_dir = lib_dir
    ocp.solver_options.model_external_shared_lib_name = lib_name

    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return acados_solver


