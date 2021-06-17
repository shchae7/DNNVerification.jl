using DNNVerification, LazySets, Test, LinearAlgebra, GLPK
import DNNVerification: ReLU, Id

acas_file = "../networks/ACASXU_run2a_1_1_tiny_4.nnet"
acas_nnet = read_nnet(acas_file, last_layer_activation = Id());

b_lower = [ -0.3242742570,  0.0318309886, -0.4999998960 ,  -0.5 ,  -0.5      ]
b_upper = [ -0.3217850849,  0.0636619772 , -0.4992041213,  -0.2272727273,  -0.1666666667      ]

in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
out_hyper  = Hyperrectangle(low = [-0.1], high = [3.0])

problem_acas1_RR = Problem(acas_nnet, in_hyper, out_hyper)

solver = BaBNeurify()
println("$(typeof(solver)) - babNeurify test")
timed_result = @timed solve(solver, problem_acas1_RR)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")