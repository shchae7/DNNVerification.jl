# Reference
# 1. https://github.com/sisl/NeuralVerification.jl/blob/457a556197a9cf34c04e8d516de21f3e3641c2f6/test/runtime/runtime_tests.jl
# 2. https://github.com/sisl/NeuralVerification.jl/blob/457a556197a9cf34c04e8d516de21f3e3641c2f6/test/runtime/runtime_aux.jl
# Acceptable forms of I/O for each tool specified in above files

using DNNVerification, LazySets, Test, LinearAlgebra, GLPK
import DNNVerification: ReLU, Id

acas_file = "../networks/ACASXU_experimental_v2a_4_5.nnet"
acas_nnet = read_nnet(acas_file, last_layer_activation = Id());

# ACAS PROPERTY 10
b_lower = [ 0.2689784272,  0.1114084602, -0.4999998960 ,  0.2272727273 ,  0.0      ]
b_upper = [ 0.6798577687,  0.4999998960 , -0.4984083465,  0.5,  0.5      ]

in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
inputSet = convert(HPolytope, in_hyper)

# output1 <= output 2
outputSet1 = HPolytope([HalfSpace([1.0, -1.0, 0.0, 0.0, 0.0], 0.0)])
problem_polytope_polytope_acas1 = Problem(acas_nnet, in_hyper, outputSet1);

# output1 <= output 3
outputSet2 = HPolytope([HalfSpace([1.0, 0.0, -1.0, 0.0, 0.0], 0.0)])
problem_polytope_polytope_acas2 = Problem(acas_nnet, in_hyper, outputSet2);

# output1 <= output 4
outputSet3 = HPolytope([HalfSpace([1.0, 0.0, 0.0, -1.0, 0.0], 0.0)])
problem_polytope_polytope_acas3 = Problem(acas_nnet, in_hyper, outputSet3);

# output1 <= output 5
outputSet4 = HPolytope([HalfSpace([1.0, 0.0, 0.0, 0.0, -1.0], 0.0)])
problem_polytope_polytope_acas4 = Problem(acas_nnet, in_hyper, outputSet4);


solver=Neurify()
println("$(typeof(solver)) - acas xu property 10")

timed_result1 =@timed solve(solver, problem_polytope_polytope_acas1)
println(" - Time: " * string(timed_result1[2]) * " s")
println(" - Output: ")
println(timed_result1[1])
println("")


timed_result2 =@timed solve(solver, problem_polytope_polytope_acas2)
println(" - Time: " * string(timed_result2[2]) * " s")
println(" - Output: ")
println(timed_result2[1])
println("")


timed_result3 =@timed solve(solver, problem_polytope_polytope_acas3)
println(" - Time: " * string(timed_result3[2]) * " s")
println(" - Output: ")
println(timed_result3[1])
println("")

timed_result4 =@timed solve(solver, problem_polytope_polytope_acas4)
println(" - Time: " * string(timed_result4[2]) * " s")
println(" - Output: ")
println(timed_result4[1])
println("")