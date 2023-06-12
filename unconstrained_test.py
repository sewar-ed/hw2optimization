import unittest
import numpy as np
from src.unconstrained_min import gradient_descent, newton, bfgs_or_sr1
from src.utils import plot_contour, plot_function_values
from tests.examples import qf_1, qf_2, qf_3, rosenbrock_f,lin_f, exp_2d
class TestUnconstrainedMin(unittest.TestCase):

    def setUp(self):
        self.methods = ["gradient_descent", "newton", "bfgs_or_sr1"]
        self.x0 = np.array([-1, 2])
        self.examples = [qf_1, qf_2, qf_3, rosenbrock_f,lin_f, exp_2d]
        self.obj_tol=10^-12
        self.param_tol=10^-8
        self.max_iter=10000
        self.flag=True

    def test_methods(self):
        for example in self.examples[3:4]:
            path_history=[]
            function_values = {}
            # Perform minimization
            _,_, history_gd = gradient_descent(example, self.x0, self.obj_tol,self.param_tol,10000,self.flag)
            #Create contour plot
            name="gradient_descent"
            path_history.append((name,history_gd))

            _,_, history_n = newton(example, self.x0, self.obj_tol,self.param_tol,self.max_iter,self.flag)
            #Create contour plot
            name="newton"
            path_history.append((name,history_n))

            
            _,_, history_bfgs = bfgs_or_sr1(example, self.x0, self.obj_tol,self.param_tol,self.max_iter,self.flag)
            #Create contour plot
            name="bfgs"
            path_history.append((name,history_bfgs))       
            
            _,_, history_sr1 = bfgs_or_sr1(example, self.x0, self.obj_tol,self.param_tol,self.max_iter,False)
            #Create contour plot
            name="sr1"
            path_history.append((name,history_sr1))



            plot_contour(example,(-2, 2),(-2, 2),path_history)



            # Store function values for plot
            # function_values["gradient_descent"] = [h[2] for h in history_gd]
            function_values["newton"] = [h[2] for h in history_n]
            function_values["bfgs"] = [h[2] for h in history_bfgs]
            function_values["sr1"] = [h[2] for h in history_sr1]

            # Create function values plot
            plot_function_values(function_values)

if __name__ == "__main__":
    unittest.main()
