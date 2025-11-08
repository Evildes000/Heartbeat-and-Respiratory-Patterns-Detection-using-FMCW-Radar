import numpy as np



class kalman_filter:
    def __init__(self, breath_NLS:float):
        """
        Initialize KF

        Parameters
        breath_NLS : estimated breathing rate
        
        """
        self.delta_t = 1 # s

        # kalman gain 1*3 array
        self.k = np.array([np.zeros(3)])
        print(f"shape of k is: {np.shape(self.k)}")
        self.state_transition = np.array([[1,self.delta_t, self.delta_t**2/2],
                                 [0, 1, self.delta_t],
                                 [0, 0, 1]], dtype=float)
        self.rou_a = 2 
        self.g = np.array([[0.5*self.delta_t ** 2],[self.delta_t], [1]], dtype=float)
        # Q refers to noise uncertainty
        self.Q = self.g @ np.transpose(self.g) * (self.rou_a ** 2)

        # observation matrix
        self.H = np.array([[1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0]])
        
        # initial state vetor
        self.state_initial = np.array([[4*breath_NLS], [0], [0]])
        # initial covariance
        self.covariance = 1000 * np.eye(3)
        self.gating = 3 # Hz
        
        self.sigma_1 = 1.5
        self.sigma_2 = 1.5
        self.sigma_3 = 1.5
        self.R = np.array([[self.sigma_1**2, 0, 0],
                  [0, self.sigma_2**2, 0],
                  [0, 0, self.sigma_3**2]], dtype=float)
        
        print(f"kalman filter is initialized")
    

    def first_update(self, estimated_heartrates:np.ndarray):
        """
        Update the parameters of KF for the first time
        
        Parameters
        ----------
        estimated_heartrates : a 3*1 array contains three estimated heartrates from ANLS

        Return
        ------
        updated_state : predicted state after the update
        updated_covariance : covariance after the update
        real_hr : the choosen heart rate
        """
        print(f"first update")
        print(f"estimated hertrates: {estimated_heartrates}")
        # predicte next state by using the current state
        # print(f"estimated_state:{estimated_heartrates}")
        predicted_state = self.state_transition @ self.state_initial
        # print(f"shape of initial state is: {np.shape(self.state_initial)}")
       # print(f"predicted_state:{predicted_state}")


        predicted_covariance = self.state_transition @ self.covariance @ np.transpose(self.state_transition) + self.Q

        # associated covariance
        ass_covariance = self.H @ predicted_covariance @ np.transpose(self.H) + self.R

        # e is measurement innovation(error)
        e = estimated_heartrates - self.H @ predicted_state
        diff = np.abs(estimated_heartrates - predicted_state)
        print(f"predicted_state is: {predicted_state}")
        print(f"e is: {e}")

        # select the NLS estimate that has min distance to the predicted state
        # predicted_error = np.abs(predicted_state - estimated_heartrates)
        print(f"shape of predicted_error: {np.shape(e)}")
        row_index = np.argmin(e)
        # if no distance smaller than the gating value
        # if len(np.where(e < self.gating)[0]) == 0:
        #    self.k = np.array([[0,0,0]])
        #    updated_state = predicted_state
        #    updated_covariance = predicted_covariance
        #    real_hr = 0
        if e[row_index] > self.gating * np.sqrt(ass_covariance[row_index, row_index]):
            self.k = np.array([[0,0,0]])
            updated_state = predicted_state
            updated_covariance = predicted_covariance
            real_hr = updated_state[0]
        # if there are estimates that close enough to predicted states, then choose the closest estimate
        else:
            print(f"row_index is: {row_index}")
            # buffer is used to calculate k
            buffer = predicted_covariance @ np.transpose(np.array([self.H[row_index]])) / ass_covariance[row_index, row_index]
            self.k[0,0] = buffer[0]
            self.k[0,1] = buffer[1]
            self.k[0,2] = buffer[2]
            # print(f"shape of k is: {np.shape(self.k)}")
            real_hr = estimated_heartrates[row_index]
            # update the state and covariance
            updated_state = predicted_state + np.transpose(self.k) * e[row_index]
            
            print(f"updated state is: {updated_state}")
            updated_covariance = (np.eye(3) - np.transpose(self.k) @ np.array([self.H[row_index]])) @ predicted_covariance

        # the returned values are used for following updates
        return [updated_state, updated_covariance, real_hr]


    def following_updates(self, estimated_heartrates, current_state, current_covariance):

        """
        This function updates the parameters of KF after the fisrt update
        
        Parameters
        ----------
        estimated_heartrates : estimated heart rates by using NLS 
        current_state : updated state from last update
        current_covariance : updated covariance from last update

        Return
        ------
        updated_state : predicted state after the update
        updated_covariance : covariance after the update
        real_hr : the choosen heart rate

        """
        # estimated_heartrates: estimated heart rates by using NLS 
        # current_state: updated state from last update
        # current_covariance: updated covariance from last update
        # this function is used to update the state after the first update
        
        print(f"estimated hertrates: {estimated_heartrates}")
        predicted_state = self.state_transition @ current_state
        # print(f"shape of current state is: {np.shape(current_state)}")
        # print(f"predicted_state:{predicted_state}")

        predicted_covariance = self.state_transition @ current_covariance @ np.transpose(self.state_transition) + self.Q

        # associated covariance
        ass_covariance = self.H @ predicted_covariance @ np.transpose(self.H) + self.R

        # e is measurement innovation(error)
        e = estimated_heartrates - self.H @ predicted_state
        print(f"e is: {e}")
        diff = np.abs(estimated_heartrates - predicted_state)
        print(f"predicted_state is: {predicted_state}")
        # print(f"diff is: {diff}")
        # row_index = np.argmin(abs(diff))
        row_index = np.argmin(abs(e))
        # select the NLS estimate that has min distance between the predicted state
        # predicted_error = np.abs(predicted_state - estimated_heartrates)
        print(f"shape of predicted_error: {np.shape(e)}")
        # print(f"error is: {predicted_error}")
        # if no estimate meet the gating value
        # if len(np.where(e < self.gating)[0]) == 0:
            # print(f"no estimates has been found")
        #    self.k = np.array([[0,0,0]])
        #    updated_state = predicted_state
        #    updated_covariance = predicted_covariance
        #    real_hr = 0
        #    print(f"heart rate is: {real_hr}")
        # if there are estimates that close enough to predicted states, then choose the closest estimate
        # if diff[row_index] > 3 * np.sqrt(ass_covariance[row_index, row_index]):
        if e[row_index] > self.gating * np.sqrt(ass_covariance[row_index, row_index]):
            self.k = np.array([[0,0,0]])
            updated_state = predicted_state
            updated_covariance = predicted_covariance
            real_hr = updated_state[0]

        else:
            # row_index = np.argmin(abs(e))
            print(f"row_index is: {row_index}")
            buffer = predicted_covariance @ np.transpose(np.array([self.H[row_index]])) / ass_covariance[row_index, row_index]
            self.k[0,0] = buffer[0]
            self.k[0,1] = buffer[1]
            self.k[0,2] = buffer[2]
            # print(f"shape of k is: {np.shape(self.k)}")
            real_hr = estimated_heartrates[row_index]
            print(f"heart rate is: {real_hr}")
            # update the state and covariance
            updated_state = predicted_state + np.transpose(self.k) * e[row_index]
            # print(f"e[row_index] is: {e[row_index]}")
            print(f"updated state is: {updated_state}")
            updated_covariance = (np.eye(3) - np.transpose(self.k) @ np.array([self.H[row_index]])) @ predicted_covariance

        # used for the next following update    
        return [updated_state, updated_covariance, real_hr]