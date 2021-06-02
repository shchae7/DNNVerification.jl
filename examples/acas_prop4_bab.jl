# Reference
# 1. https://github.com/sisl/NeuralVerification.jl/blob/457a556197a9cf34c04e8d516de21f3e3641c2f6/test/runtime/runtime_tests.jl
# 2. https://github.com/sisl/NeuralVerification.jl/blob/457a556197a9cf34c04e8d516de21f3e3641c2f6/test/runtime/runtime_aux.jl
# Acceptable forms of I/O for each tool specified in above files

using DNNVerification, LazySets, Test, LinearAlgebra, GLPK
import DNNVerification: ReLU, Id

acas_file = "../networks/ACASXU_run2a_4_5_batch_2000_out1.nnet"
acas_nnet = read_nnet(acas_file, last_layer_activation = Id());

# ACAS PROPERTY 4
# From Ref #2
b_lower = [ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.4      ]
b_upper = [ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.4      ]

in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
out_hyper  = Hyperrectangle(low = [-0.1], high = [3.0])

problem_acas1_RR = Problem(acas_nnet, in_hyper, out_hyper)

solver = BaB()
println("$(typeof(solver)) - acas")
timed_result = @timed solve(solver, problem_acas1_RR)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")