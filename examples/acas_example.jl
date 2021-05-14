using BaBNeurify, LazySets, Test, LinearAlgebra, GLPK
import BaBNeurify: ReLU, Id

acas_file = "../networks/tiny_nnet.nnet"
acas_nnet = read_nnet(acas_file, last_layer_activation = Id());

# ACAS PROPERTY 10 - modified
# Original input range: 
# LOWER BOUND: array([[ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.15      ]])
# UPPER BOUND: array([[ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.65      ]])

b_lower = [ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.4      ]
#b_upper = [ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.4      ]
b_upper = [ 0.3,  0.2 , -0.49840835,  0.3920202,  0.4      ]

in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
inputSet = convert(HPolytope, in_hyper)

# output1 <= output 5
outputSet = HPolytope([HalfSpace([1.0, 0.0, 0.0, 0.0, -1.0], 0.0)])

problem_polytope_polytope_acas = Problem(acas_nnet, in_hyper, outputSet);

solver=NeurifyBaB()
println("$(typeof(solver)) - acas")
timed_result =@timed solve(solver, problem_polytope_polytope_acas)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")