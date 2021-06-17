using DNNVerification, LazySets, Test, LinearAlgebra, GLPK
import DNNVerification: ReLU, Id

acas_file = "./networks/small_nnet.nnet"
acas_nnet = read_nnet(acas_file, last_layer_activation = Id());

b_lower = [ 1 ]
b_upper = [ 2 ]

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