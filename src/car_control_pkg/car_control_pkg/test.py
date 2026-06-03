import casadi as ca
import numpy as np
import time

# --------------------------------------------------------------------------
# --- 1. MPC & System Parameters -------------------------------------------
# --------------------------------------------------------------------------
# --- Prediction Horizon ---
# How many steps the MPC looks into the future.
PREDICTION_HORIZON = 20  # N (steps)

# --- Timesteps ---
# The "big" step for the MPC controller (10 Hz)
MPC_TIMESTEP = 0.1  # dt (seconds)
# The "small" step for the physics model (100 Hz)
MODEL_TIMESTEP = 0.01 # dt_nn (seconds)

# --- Dynamics ---
# How many small model steps fit into one big MPC step
INTEGRATION_STEPS = int(MPC_TIMESTEP / MODEL_TIMESTEP) # = 10 steps
STATE_DIM = 4         # [x, z, sin(angle), cos(angle)]
CONTROL_DIM = 4       # [v_fl, v_fr, v_rl, v_rr]

# --- Vehicle & Cost Function Parameters ---
TARGET_AVG_VELOCITY = 2.0  # m/s (The "Go Fast" target)
MAX_WHEEL_VELOCITY = 2.5   # Max physical speed of any one wheel
MIN_WHEEL_VELOCITY = -0.5  # Min physical speed (e.g., reverse)

# --------------------------------------------------------------------------
# --- 2. Placeholders (Replace with your actual functions) -----------------
# --------------------------------------------------------------------------

def load_continuous_time_ab_model():
    """
    CRITICAL: This function must load your model and wrap it as a
    CasADi-compatible function that returns the *continuous-time* A and B.
    
    This is a FAKE symbolic model. REPLACE THIS.
    """
    print("Loading (fake) Continuous-Time A, B NN model...")
    
    state_in = ca.MX.sym('state_in', STATE_DIM)
    control_in = ca.MX.sym('control_in', CONTROL_DIM)
    
    # Fake Dynamics: A = Identity, B = Identity
    # This implies dot_x = 1*x + 1*u, just for placeholder.
    # Your NN will return complex, state-dependent matrices.
    A_matrix = ca.MX.eye(STATE_DIM)
    B_matrix = ca.MX.eye(CONTROL_DIM) # (Assumes B is 4x4)
    
    # Create a CasADi Function
    nn_model_func = ca.Function(
        'continuous_ab_model',
        [state_in, control_in],
        [A_matrix, B_matrix],
        ['state_in', 'control_in'],
        ['A_out', 'B_out']
    )
    return nn_model_func

def get_car_state_from_unity():
    """
    Placeholder: Get the car's current state from Unity.
    MUST return: [x, z, sin(angle), cos(angle)]
    """
    # Example: At x=1, z=2, angle=0
    return np.array([1.0, 2.0, 0.0, 1.0]) 

def get_reference_path(all_path_points, current_state, horizon, timestep):
    """
    Placeholder: Get the future reference path [x, z] coordinates
    for the next 'horizon' steps.
    """
    target_path = np.zeros((horizon + 1, 2)) # (N+1, 2)
    current_pos = current_state[:2] # Get [x, z] from state
    
    # This logic should be much smarter:
    # 1. Find closest point on 'all_path_points'
    # 2. Sample N+1 points along the path
    for i in range(horizon + 1):
        # Fake path: move 1.0 m/s in x-direction
        target_path[i, :] = current_pos + [i * timestep * 1.0, 0] 
    return target_path

def send_control_to_unity(wheel_velocities):
    """
    Placeholder: Send the calculated 4 wheel velocities to Unity.
    """
    # wheel_velocities is a 4-element np.array [v_fl, v_fr, v_rl, v_rr]
    # print(f"Sending controls: {wheel_velocities}")
    pass

# --------------------------------------------------------------------------
# --- 3. MPC Solver Creation Function --------------------------------------
# --------------------------------------------------------------------------

def create_mpc_solver(continuous_ab_model):
    """
    Creates the CasADi NLP solver object.
    """
    print("Creating MPC solver...")
    opti = ca.Opti() # The optimization problem

    # --- Decision Variables (The "Unknowns") ---
    
    # The solver will find the optimal values for these:
    # state_trajectory: The predicted state [x,z,sin,cos] over the horizon
    state_trajectory = opti.variable(PREDICTION_HORIZON + 1, STATE_DIM)
    # control_trajectory: The optimal controls [v_fl,..] over the horizon
    control_trajectory = opti.variable(PREDICTION_HORIZON, CONTROL_DIM)

    # --- Parameters (The "Knowns") ---
    
    # We will provide these values in every loop:
    # current_state_param: The car's state *right now*
    current_state_param = opti.parameter(STATE_DIM)
    # reference_path_param: The desired path [x,z] to follow
    reference_path_param = opti.parameter(PREDICTION_HORIZON + 1, 2)

    # --- Cost Function (J) (What we want to minimize) ---
    cost = 0
    
    # Cost function weights (TUNE THESE)
    state_cost_weight = 100.0         # Q: Penalty for path error
    control_smoothness_weight = 1.0 # R: Penalty for jerky acceleration
    target_velocity_weight = 10.0     # W: Penalty for driving too slow

    for k in range(PREDICTION_HORIZON):
        # 1. Path Tracking Cost (Cost for being off the path)
        predicted_pos = state_trajectory[k, 0:2] # Get [x, z]
        reference_pos = reference_path_param[k, :]
        position_error = predicted_pos - reference_pos
        cost += state_cost_weight * ca.dot(position_error, position_error)

        # 2. "Go Fast" Cost (Cost for being below target velocity)
        predicted_avg_vel = ca.sum1(control_trajectory[k, :]) / CONTROL_DIM
        velocity_error = TARGET_AVG_VELOCITY - predicted_avg_vel
        cost += target_velocity_weight * (velocity_error**2)

        # 3. Control Smoothness Cost (Cost for jerky acceleration)
        if k > 0:
            predicted_accel = (control_trajectory[k, :] - control_trajectory[k - 1, :]) / MPC_TIMESTEP
            cost += control_smoothness_weight * ca.dot(predicted_accel, predicted_accel)
            
    # Add final state error (cost for being off path at the end)
    final_pos = state_trajectory[PREDICTION_HORIZON, 0:2]
    final_target = reference_path_param[PREDICTION_HORIZON, :]
    final_error = final_pos - final_target
    cost += state_cost_weight * ca.dot(final_error, final_error)

    opti.minimize(cost)

    # --- Constraints (The "Rules" of the simulation) ---
    
    # Dynamics Constraint (How the car moves)
    # This is the "hybrid" approach
    for k in range(PREDICTION_HORIZON):
        # The "big" MPC step (0.1s)
        current_state_in_loop = state_trajectory[k, :]
        current_control_in_loop = control_trajectory[k, :]
        
        # Integrate 10 times using the 0.01s NN model
        for i in range(INTEGRATION_STEPS):
            
            # --- THIS IS YOUR NEW MODEL ---
            # 1. Get continuous-time A and B from the NN
            A_matrix, B_matrix = continuous_ab_model(current_state_in_loop, current_control_in_loop)
            
            # 2. Calculate the state derivative: dot_x = A*x + B*u
            state_col = current_state_in_loop.T
            control_col = current_control_in_loop.T
            
            state_derivative = ca.mtimes(A_matrix, state_col) + ca.mtimes(B_matrix, control_col)
            
            # 3. Forward Euler: x_next = x_now + dot_x * dt
            state_col_next = state_col + state_derivative * MODEL_TIMESTEP
            
            # Update state for the next integration loop
            current_state_in_loop = state_col_next.T
        
        # The result of 10 small steps must equal the next MPC state
        opti.subject_to( state_trajectory[k + 1, :] == current_state_in_loop )

    # Initial State Constraint
    # The *first* predicted state must be the *actual* current state
    opti.subject_to( state_trajectory[0, :] == current_state_param.T )

    # Control Limits Constraints
    # All 4 wheel velocities must be within min/max bounds
    opti.subject_to( opti.bounded(MIN_WHEEL_VELOCITY, 
                                 control_trajectory, 
                                 MAX_WHEEL_VELOCITY) )

    # --- Create the Solver ---
    solver_options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3}
    opti.solver('ipopt', solver_options)
    
    print("Solver created.")
    return opti, state_trajectory, control_trajectory, current_state_param, reference_path_param


# --------------------------------------------------------------------------
# --- 4. Main Control Loop -------------------------------------------------
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Load your NN model once
    nn_ab_model = load_continuous_time_ab_model()
    
    # 2. Create the MPC solver once
    (solver, 
     state_traj_vars, 
     control_traj_vars, 
     current_state_p, 
     reference_path_p) = create_mpc_solver(nn_ab_model)
    
    # 3. Get your list of all path points
    all_path_points = np.zeros((100, 2)) # Load your real path
    
    print("Starting MPC control loop...")
    try:
        while True:
            loop_start_time = time.time()
            
            # --- Step 1: Get Current State ---
            current_state = get_car_state_from_unity() # [x, z, sin, cos]
            
            # --- Step 2: Get Future Target Path ---
            reference_path = get_reference_path(
                all_path_points, current_state, PREDICTION_HORIZON, MPC_TIMESTEP
            )
            
            # --- Step 3: Set Parameters ---
            solver.set_value(current_state_p, current_state)
            solver.set_value(reference_path_p, reference_path)

            # --- Step 4: Solve the NLP ---
            try:
                solution = solver.solve()
                # Get the entire optimal control plan
                optimal_controls = solution.value(control_traj_vars)
                # We only apply the *first* step of the plan
                control_to_apply = optimal_controls[0, :] 
                
            except Exception as e:
                print(f"Solver failed! {e}")
                control_to_apply = np.zeros(CONTROL_DIM) # Fallback: stop

            # --- Step 5: Send Control to Unity ---
            send_control_to_unity(control_to_apply)
            
            # --- Step 6: Wait for next cycle ---
            elapsed_time = time.time() - loop_start_time
            sleep_time = MPC_TIMESTEP - elapsed_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: MPC loop took {elapsed_time:.3f}s (Budget: {MPC_TIMESTEP}s)")

    except KeyboardInterrupt:
        print("Stopping MPC loop.")
        send_control_to_unity(np.zeros(CONTROL_DIM))