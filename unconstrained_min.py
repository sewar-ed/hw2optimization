import numpy as np
def gradient_descent(f,x0,obj_tol, param_tol,max_iter,flag):
    alpha=0.1
    path=[]
    x_prev=x0
    f_prev=f.evaluate(x0)
    df_prev=f.eval_grad(x0)
    i=0
    success=False
    while not success and i<max_iter:
        path.append((i,x_prev,f_prev)) #append i, curr_x and curr_f for later use.
        #line search with wolfe condition
        alpha,x_next,f_next,df_next=wolfe_search(alpha,x_prev,f,f_prev,df_prev,0.5, 0.01)
        print(f'x:{x_prev}, xnext:{x_next}, f_prev:{f_prev}, f_next:{f_next}')
        i+=1
        success=check_converge(obj_tol,param_tol,x_next,x_prev,f_next,f_prev)   
        x_prev=x_next
        f_prev=f_next
        df_prev=df_next
    return x_next, success, path


def newton(f,x0, obj_tol,param_tol,max_iter,flag):
    alpha=1
    #init params to use in calc.
    path=[]
    x_prev=x0
    x_next=x0
    p_k=0
    f_prev=f.evaluate(x_prev)
    df_prev=f.eval_grad(x_prev)
    success=False
    i=0
    while not success and i<max_iter:
        f_next=f.evaluate(x_next)
        df_next=f.eval_grad(x_next)
        f_hess=f.eval_hess(x_next)
        #check convergence
        if np.linalg.norm(f_next) < obj_tol:
            success=True
        #use numpy to solve linear system
        if np.any(f_hess != 0):
            p_k=np.linalg.solve(f_hess, -df_next)
        else:
            p_k=-df_next
        path.append((i,x_next,f_next)) #append iteration info
        # Perform line search with Wolfe condition
        alpha,_,_,_=wolfe_search(alpha,x_prev,f,f_prev,df_prev, 0.5, 0.01)
        x_next=x_prev-p_k*alpha
        i+=1
        x_prev=x_next
        f_prev=f_next
        df_prev=df_next
    return x_next,success,path



def bfgs_or_sr1(f,x0, obj_tol,param_tol,max_iter,flag):
    alpha=1
    path=[]
    i=0
    f_prev=f.evaluate(x0)
    df_prev=f.eval_grad(x0)
    x_prev=x0
    x_next=x0
    #hessian_prev=np.eye(len(x0)) #init hessian
    hessian_next=np.eye(len(x0))
    success=False
    while not success and i<max_iter:
        f_next=f.evaluate(x_next)
        df_next=f.eval_grad(x_next)
        #check convergence
        success=check_converge(obj_tol,param_tol,x_next,x_prev,f_next,f_prev)
        #use numpy to solve linear system
        p_k=np.linalg.solve(hessian_next, -df_next)
        
        alpha,x_next,f_next,df_next=wolfe_search(alpha,x_prev,f,f_prev,df_prev,bt=0.5, cc=0.01)
        step_v=alpha*p_k #step vector
        x_next=x_prev+ step_v
        df_next=f.eval_grad(x_next)
        diff_grd=df_next-df_prev #difference between gradients
        if flag: #if bfgs update hessian accordingly
            r=1/np.dot(step_v.T,diff_grd) #calc reciprocal
            li = (np.eye(len(x0))-(r*((step_v@(diff_grd.T))))) #left intermidiate 
            ri = (np.eye(len(x0))-(r*((diff_grd@(step_v.T))))) #right intermidiate
            hess_inter = li@hessian_next@ri #intermidiate hessian
            hessian_next= hess_inter + (r*((step_v@(step_v.T)))) 
        else: #if sr1 update hessian accordingly
            hess_step=hessian_next@step_v
            diff_grd_hess=diff_grd- hess_step
            hess_inter=hessian_next + np.outer(diff_grd_hess, diff_grd_hess) / np.dot(diff_grd_hess,step_v)
            hessian_next=hess_inter - np.outer(hess_step,hess_step) / np.dot(step_v,hess_step)
        i+=1
        path.append((i,x_next,f_next)) #append iteration info
        x_prev=x_next
        df_prev=df_next
        f_prev=f_next
    return x_next, success,path

def check_converge(obj_tol,param_tol,x_next,x_prev,f_next,f_prev)->bool:
    #check if we meet successful termination criterion.
    if f_prev-f_next <=obj_tol or np.all(np.abs(x_next-x_prev)<=param_tol):
        return True
    return False

def wolfe_search(alpha,x_prev,f,f_prev,df_prev, bt=0.5, cc=0.01):
    alpha_new=alpha
    x_next=x_prev-alpha_new*df_prev
    f_n=f.evaluate(x_next)
    df_n=f.eval_grad(x_next)
    while f_n > f_prev + bt * alpha_new * np.dot(df_prev, alpha_new*df_prev) or np.dot(df_n, alpha_new*df_prev) > cc * np.dot(df_prev, alpha_new*df_prev):
        alpha_new /= 2  # Reduce step size
        x_next = x_prev - alpha_new * df_prev
        # if np.all(x_next== x_prev): count+=1
        # if count >10: break
        f_n = f.evaluate(x_next)
        df_n = f.eval_grad(x_next)
    return alpha_new,x_next,f_n,df_n 