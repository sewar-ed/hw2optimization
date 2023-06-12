import numpy as np
class quad:
    
    def __init__(self,q_matrix) -> None:
        self.q_matrix=q_matrix
        self.grad_coeff=q_matrix + q_matrix.T
    
    def evaluate(self,x):
        x=np.array(x,dtype=np.float64)
        return x.T@ self.q_matrix @ x
    
    def eval_grad(self,x):
        x=np.array(x,dtype=np.float64)
        return self.grad_coeff@x

    def eval_hess(self,_):
        return self.grad_coeff
    
#init instances of quadratic objective functions
q_1=np.array([[1,0],[0,1]])
qf_1=quad(q_1)

q_2=np.array([[1,0],[0,100]])
qf_2=quad(q_2)

q_3_sub=np.array([[np.sqrt(3)/2, -1/2],[1/2,np.sqrt(3)/2]], dtype=float)
q_3=q_3_sub.T@ np.flip(q_2) @ q_3_sub
qf_3=quad(q_3)


#rosenbrock function implementation
class rosenbrock():
        def evaluate(self,x):
            eval_x=100*(x[1]- x[0]**2)**2 + (1-x[0])**2
            return eval_x 
        
        def eval_grad(self,x):
            df_dx = -400*x[0] * (x[1] - x[0]**2) - 2* (1 - x[0])
            df_dy = 200* (x[1] - x[0]**2)
            #print(f'grad:{df_dx,df_dy}')
            return np.array([df_dx, df_dy])
        
        def eval_hess(self,x):
            df_dx_dx=-400*(x[1] - 3*x[0]**2) +2
            df_dx_dy=-400*x[0]
            df_dy_dy=200
            return np.array([[df_dx_dx,df_dx_dy],[df_dx_dy,df_dy_dy]])

        
rosenbrock_f=rosenbrock()

    #linear function 
class linear_f():
    def __init__(self,a)->None:
        self.a=a

    def evaluate(self,x):
        x=np.asarray(x,dtype=np.float64)
        return self.a.T @ x
    
    def eval_grad(self,x):
        return self.a
    
    def eval_hess(self,x):
        return np.zeros((len(self.a),len(self.a)))

lin_f=linear_f(np.array([1,2]))
    
    #The last function 
class exponent_2d():
    
    def evaluate(self,x):
        fst_exp= x[0] + 3*x[1]- 0.1
        snd_exp=x[0] - 3*x[1]-0.1
        thd_exp=-x[0]-0.1
        return np.sum(np.exp([fst_exp,snd_exp,thd_exp]))
    
    def eval_grad(self,x):
        fst_exp= x[0] + 3*x[1]- 0.1
        snd_exp=x[0] - 3*x[1]-0.1
        thd_exp=-x[0]-0.1
        df_1=np.sum(np.exp([fst_exp,snd_exp]))-np.exp(thd_exp)
        df_2=np.exp(fst_exp) -3* np.exp(snd_exp)
        return np.array([df_1,df_2])
    
    def eval_hess(self,x):
        fst_exp= x[0] + 3*x[1]- 0.1
        snd_exp=x[0] - 3*x[1]-0.1
        thd_exp=-x[0]-0.1
        df_dx_dx=np.sum(np.exp([fst_exp,snd_exp,thd_exp]))
        df_dx_dy=3*np.exp(fst_exp)-3*np.exp(snd_exp)
        df_dy_dy=9*np.exp(fst_exp)+9*np.exp(snd_exp)
        return np.array([[df_dx_dx,df_dx_dy],[df_dx_dy,df_dy_dy]])

exp_2d=exponent_2d()