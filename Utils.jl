using LinearAlgebra
using Base.Iterators;
using Combinatorics
using Printf
using SymPy
using Optim 
using Symbolics
using QuadGK


include("Run_configs.jl")
include("Quantum_gates.jl")


# Definition of Pauli matrices
sx= [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]
p_0 = diagm([1.0,0.0])
p_1 = diagm([0.0,1.0])
v0 = [1.0, 0.0]
v1 = [0.0, 1.0]


# Returns the computational basis vector
comp_basis(i, num_qubit) = diagm(one_at(i,2^num_qubit))

# Converts decimal number to binary array (+1,-1) based on number of qubits
function sign_arr(ind, num_qubit)
    arr = digits(ind-1, base=2, pad=num_qubit) |> reverse
    return 2* arr .- 1
end


ZZ_coef = [1,-1, -1, 1]
# test if the input matrix is diagonal
function is_diagonal(mat)
    # This function tests if the input matrix is diagonal
    coords = findall(i -> (abs(real(i)) > 0.01 || abs(imag(i)) > 0.01),mat)
    for i in coords
        if i[1] != i[2]
            return false
        else
            continue
        end
    end 
    return true
end


function merge_arr(array_of_arrays)
    # This function merges an array of arrays into a single array
    to_return = []
    for i in array_of_arrays
        to_return = append!(to_return, i)
    end
    return to_return
end


function is_in(val, list, precision)
    # This function tests if an eigenvalue is already present in the storage list
    for (i,x) in enumerate(list)
        if abs(round(x,digits=precision) - round(val,digits=precision)) < 1e-8
            return true, i
        end
    end
    return false, 0
end

function rotation_decomposition(U)
    # This function decomposes any arbitrary single qubit unitary into three elementary rotations 
    Y_4 = exp(im * pi/4 * sy)
    U_pr = Y_4 * U * Y_4'
    a,b,c,d = U_pr[1,1],U_pr[1,2],U_pr[2,1], U_pr[2,2]
    @assert sum(abs.([a*d, c*d, b*d, a * b, a * c ,b*c])) != 0

    c1_offsets = [0, pi, -pi]
    c2_mults = [1, -1]

    for (c1_offset, c2_mult) in Iterators.product(c1_offsets, c2_mults)
        c1 = log(-a*b/(c*d))/(-2 * im) + c1_offset
        c3 = log(-a*c/(b*d))/(-2 * im)
        c2 = 2 * c2_mult * atan(sqrt(-b*c/(a*d)))
        # println(c1," ",c2," ", c3)
        U_pr_re = exp(-im * c1/2 * sz) * exp(-im * c2/2 * sy) * exp(-im * c3/2 * sz)
        fid = abs(tr(U_pr' * U_pr_re) / tr(U_pr' * U_pr))
        if abs(1-fid) < 1e-5
            constructed_U = exp(-im * c1/2 * sx) * exp(-im * c2/2 * sy) * exp(-im * c3/2 * sx)
            fidelity =  abs(tr(U' * constructed_U) / tr(U' * U))
            if abs(1-fidelity) < 1e-5
                println("Fidelity: ", fidelity)
                return real(-c1/2),real(-c2/2),real(-c3/2)
            end
        end
    end
    println("Solution not found - :(")
    return 0,0,0            
end


function get_m_squared(g, is_l, dim)
    # This function calculates M^2 from G
    # If we are in the l subalgebra, the involution we choose is sigma_x_n
    # If we are not, the involution is sigma_z_n
    if is_l
        invo = kron(I(2^(dim-1)),sx)
    else
        invo = kron(I(2^(dim-1)),sz)
    end
    return invo * g' * invo * g
end

function is_Unitary(mat,dim)
    # This function tests if the given input matrix is unitary
    return sum(abs.(mat' * mat .- I(2^dim))) < error_threshold
end

function get_sigma_n(if_x, dim)
    # This function returns the involution operator, sigma_nx or sigma_nz depending on the stage of decomposition
    if if_x
        return kron(I(2^(dim-1)) , sx)
    else
        return kron(I(2^(dim-1)) , sz)
    end
end

# The list of constant gates considered for heuristics during decomposition
single_qubit_gates = [R_x(pi/2),R_x(pi/4), R_x(pi), R_y(pi/2), R_y(pi/4),R_y(pi), get_H()]

two_qubit_gates = [get_C_NOT(), get_C_Z(),get_SWAP()]


function all_good(candidate,σzₙ)
    N = length(candidate)
    # @assert N > 1
    for i in 1:N
        for j in i+1:N
            if abs(dot(candidate[i], candidate[j])) > error_threshold
                println("non-orthogonality of basis vectors: ",abs(dot(candidate[i], candidate[j])))
            end
            if abs(dot(normalize(candidate[i].+ σzₙ * candidate[i]),normalize(candidate[j].+ σzₙ * candidate[j]))) > error_threshold
                return false
            else
            end
        end
    end
    return true
end

function find_basis_legacy(normal_list,size,σzₙ)
    # Note: This is the legacy version of a function
    is_found = false
    for (i,v) in enumerate(normal_list)
        compat_list = [v]
        for candid in normal_list[i+1:end]
            if abs(candid'* σzₙ * v) < error_threshold 
                push!(compat_list, candid)
            end
        end
        if !(length(compat_list) >= size)
            println(i," doesn't have enough compatible pairs")
            continue
        end
        for trial_set in collect(combinations(compat_list, size))
            if all_good(trial_set, σzₙ)
                println("Solution found!! Yayy!")
                return true, trial_set
            end
        end
    end
    println("No suitable basis found, this is very bad.")
    return false, []     
end


function has_pair(eigenval_list, val)
    # Function returns the index of the unpaired 
    for (ind,e) in enumerate(eigenval_list)
        if round(conj(e),digits=2) == round(val,digits=2)
            if ((ind > 1 ? round(eigenval_list[ind-1],digits=2) : 0) != round(val,digits=2)) && (ind == length(eigenval_list) || round(eigenval_list[ind+1],digits=2) != round(val,digits=2)) 
                return ind+1
            end
        end
    end
    # Now it means we are unpaired at all, append in the end
    return 0
end

function construct_basis(input_list, size, σzₙ,dim, pairs_list)
    # This function constructs a proper orthogonal basis from p to q that satisfy the constraints in appendix A in the paper
    if !isempty(pairs_list)
        pool = filter!(x->sum(abs.([dot(x' * σzₙ * pair) for pair in pairs_list]))< error_threshold,input_list)
    else
        pool = input_list
    end
    len_pool = length(pool)
    V = Matrix{ComplexF64}(undef, 2^dim, len_pool)
    for i in 1:len_pool
        V[:,i] = pool[i]
    end
    K = abs.(V'* σzₙ * V)
    remained_list = []
    for ind in 1:len_pool
        compat_len = length(findall(x-> abs(x) < error_threshold, K[:,ind]))
        if compat_len >= size - 1
            push!(remained_list,ind)
        end
    end
    println("Length of valid candidates: ",length(remained_list))
    if print_verbose
        display(K)
    end
    K_red = K[remained_list, remained_list]
    rem_len = length(remained_list)
    for i in 1:rem_len
        println("Now processing ", i, " th index")
        for trial_indices in collect(combinations(findall(x -> abs(x) < error_threshold,K_red[i,i+1:end]), size-1))
            trial_indices = trial_indices .+ i
            K_trial = K_red[trial_indices, trial_indices]
            for i in 1:size-1
                K_trial[i,i] = 0
            end
            if sum(K_trial) < error_threshold
                println("Solution found with ", length(trial_indices) + 1, " basis vectors and they are all orthogonal.")
                return true, pool[remained_list[[[i];trial_indices]]]
            end
        end
        println("This doesn't work")
    end
    println("No orthogonal basis can be found.")
    return false,[]
end



function get_log(matrix)
    # This function returns log(matrix)
    round_matrix = round.(matrix, digits=30)
    spectral_decomp = eigen(round_matrix)
    liealg_element = spectral_decomp.vectors * Diagonal(log.(Complex.(spectral_decomp.values))) * inv(spectral_decomp.vectors)
    return liealg_element
end

# The following are Lie algebra elements of su(4)
su_4_1 = [I(4), kron(sx, I(2)), kron(sy,I(2)), kron(sz,I(2))]
su_4_2 = [I(4), kron(I(2),sx), kron(I(2),sy), kron(I(2),sz)]

function get_coef_2(gate, return_m)
    # This function returns the coefficient of each lie algbera element under the Pauli basis for the input gate
    g_m = zeros(ComplexF64,4,4)
    coef = zeros(ComplexF64, 4, 4)
    for (i, j) in Iterators.product(1:4, 1:4)
        candidate = su_4_1[i] * su_4_2[j]
        coef[i,j] = 1/4 * tr(candidate * gate)
        if return_m && (i != 1 && j!=1)
            g_m += coef[i,j] * candidate
        end
    end
    # coef[1,1] = 0 # Neither SU4 nor SU8 includes the identity tensor product
    if return_m
        return coef, g_m
    else
        return coef
    end
end

function solve_m(g::Matrix{ComplexF64}, g_m::Matrix{ComplexF64})
    N = size(g, 1)

    # complex matrix indeces for N x N matrix
    m = [ Sym("m_$(i)_$(j)") for i in 1:N, j in 1:N ]

    function comm(A, B)
        return A*B - B*A
    end

    eq = m - g_m + comm(g, m)/2 - comm(m, comm(m, g))/4

    # convert to vectors for nsolve
    eq_v = vec(eq)
    m_v = vec(m)

    # solve numerically for m
    sol_v = nsolve(eq_v, m_v, [0.0 for i in 1:N^2])

    # convert to matrix
    sol = reshape(sol_v, N, N)

    return convert(Matrix{ComplexF64}, sol)
end

# Prints coefficient of Pauli basis for given coefficient table
pauli_dict = Dict(1=>"I", 2=>"X", 3=>"Y", 4=>"Z")
function print_pauli(coef,n_dim)
    coord = findall(i -> (abs(real(i)) > 0.01 || abs(imag(i)) > 0.01),coef)
    println("The pauli coefficients are:")
    for i in coord
        if n_dim == 3
            @printf("%s%s%s: %8.2f+%8.2fim \n",get(pauli_dict, i[1],3),get(pauli_dict, i[2],3), get(pauli_dict, i[3],3), real(coef[i]), imag(coef[i]))
        elseif n_dim == 2
            @printf("%s%s: %8.2f+%8.2fim \n",get(pauli_dict, i[1],3),get(pauli_dict, i[2],3), real(coef[i]), imag(coef[i]))
        end
    end
end

function trace_over(gate, dim, dims_target)
    # Input dims_from_last: dimension of desired traced over operator
    while dim > dims_target
        trace_10 = kron(I(2^(dim-1)) ,[1, 0])
        trace_01 = kron(I(2^(dim-1)) ,[0, 1])
        gate = (trace_01' * gate * trace_01 .+ trace_10' * gate * trace_10)/2
        dim -= 1
    end
    return gate
end

function trace_from(gate,dim,dims_target)
    # Returns the matrix with the first dimensions traced over until its dimension is dims_target
    while dim > dims_target
        trace_10 = kron([1, 0],I(2^(dim-1)))
        trace_01 = kron([0, 1],I(2^(dim-1)))
        gate = (trace_01' * gate * trace_01 .+ trace_10' * gate * trace_10)/2
        dim -= 1
    end
    return gate
    
end

function trace_at(mat,dim,ind)
    # This function traces the ind^th qubit subspace over 
    proj_0 = kron(I(2^(ind-1)),kron(v0,I(2^(dim - ind))))
    proj_1 = kron(I(2^(ind-1)),kron(v1,I(2^(dim - ind))))
    return (proj_0' * mat * proj_0 + proj_1' * mat * proj_1)/2
end


function trace_away(list_ind,mat, dim)
    # This function takes a list of indices at which the Hilbert space should be traved over
    # Here we need to assert that the input list of indices is in ascending order
    to_return = mat
    actual_ind = [ind - order + 1 for (order, ind) in enumerate(list_ind)]
    for (i, ind) in enumerate(actual_ind)
        to_return = trace_at(to_return, dim - i+1, ind)
    end
    return to_return
end


# This function detects whether a given unitary matrix can be separated into product of 
# Unitary of smaller dimensions, i.e. separable operation
function reduce_to_product(input, dimens)
    to_be_decomposed = [(input,dimens)]
    product = []
    while length(to_be_decomposed) > 0
        (mat,dim) = pop!(to_be_decomposed)
        converse_list = []
        for i in 1:2^dim
            if i in converse_list
                continue
            else
                mask = reverse(digits(i, base=2, pad = dim))
                A = trace_at(mat,dim, mask)
                B = trace_at(mat,dim, 1 .- mask)
                if sum(abs.(A*B .- mat)) < error_threshold
                    # We have succeeded a decomposition and now can process further
                    push!(to_be_decomposed, (A,sum(mask)))
                    push!(to_be_decomposed,(B,dim-sum(mask)))
                    break
                end
                push!(converse_list,2^dim-1-i)
            end
        end
        # if by then we haven't gotten to a break, it means the matrix is not decomposable, put it in the product list simply
        push!(product,(mat,dim))
    end

    return product
end



function skip_N(mat :: Matrix{ComplexF64},dim, ind,N)
    # This function computes the unitary representing U(i, i+N) in a multiqubit network, N is the number of qubits skipped
    # input mat is the unitary
    # dim is the dimension of the input unitary mat
    # N is the number of additional skipped over qubits
    # ind is the starting location of after which qubit to skip
    ret = Matrix{ComplexF64}(undef, (2^(dim+N), 2^(dim+N)))
    D = 2^(dim+N-1)
    # reminder of how big to slice the original unitary
    if N == 0
        return mat
    end
    if ind == 0
        return kron(I(2),mat)
    end
    if ind == dim
        return kron(mat,I(2))
    end
    M = 2^(dim-1)
    if ind == 1
        for (i,j) in [(1,1),(1,2),(2,1),(2,2)]
            ret[(i-1)*D+1 : i*D, (j-1)*D+1 : j * D]=kron(I(2^N), mat[(i-1)*M+1 : i*M, (j-1)*M+1 : j*M])
        end
    else
        for (i,j) in [(1,1),(1,2),(2,1),(2,2)]
            ret[(i-1)*D+1 : i*D, (j-1)*D+1 : j * D]=skip_N(mat[(i-1)*M+1 : i*M, (j-1)*M+1 : j*M], dim-1, ind-1, N)
        end
    end
    return ret
end
      


function get_total_unitary(mat, total_dim,list_qubit)
    # This function takes a unitary mat as input at dimension M
    # Total dim is the total number of qubits in the system
    # list_qubit is the list of participating qubits
    unit = mat
    dim = length(list_qubit)
    complete_list = list_qubit[1]:last(list_qubit)
    ind = 0
    for i in complete_list
        if !(i in list_qubit)
            # then this qubit is a participating qubit and will not be skipped
            unit = skip_N(unit,dim,ind,1)
            dim += 1
        end
        ind+=1
    end
    return kron(I(2^(list_qubit[1]-1)),kron(unit, I(2^(total_dim-last(list_qubit)))))
end


function separation(target_gate,list_qubit,dim)
    # First step: Generate all the partitions of the dimensions into TWO synthesize
    if dim == 1
        return [(target_gate, list_qubit,dim)]
    end
    separated_list = []
    for partition in partitions(1:dim, 2)
        # Every such partition is evaluated to be a list of 2 lists
        traced_1 = trace_away(partition[2], target_gate,dim)
        traced_2 = trace_away(partition[1], target_gate,dim)
        if sum(abs.(get_total_unitary(traced_1,dim,partition[1]) * get_total_unitary(traced_2,dim,partition[2]) .- target_gate)) < error_threshold
            append!(separated_list,separation(traced_1,[list_qubit[j] for j in partition[1]],length(partition[1])))
            append!(separation(traced_2,[list_qubit[j] for j in partition[2]],length(partition[2])))
            println("Unitary is separable.")
            return separated_list
        end
    end
    return [(target_gate,list_qubit,dim)]
end

function detect_reduction(mat,dim)
    # This function scans through the qubit dimensions of the unitary and detects non participation qubits
    nonpar_list = []
    for ind in 1:dim
        if sum(abs.(mat .- skip_N(trace_at(mat,dim,ind),dim-1,ind-1,1))) < error_threshold
            push!(nonpar_list, ind)
        end
    end
    if !isempty(nonpar_list)
        println("Dimension reduced!!")
    end
    return nonpar_list
end


function participating_qubits(traced_over_list,list_qubit)
    # Input traced_over_list is the list of indices of list_qubit that is not participating
    # This function filters the list_qubit to not include the traced_over_list elements
    to_return = []
    for (ind,i) in enumerate(list_qubit)
        if ind in traced_over_list
            continue
        else
            push!(to_return,i)
        end
    end
    return to_return
end


function dim_reduction(mat,dim,list_qubit)
    # This function aims to reduce the dimension of the matrix to be decomposed as part of the heuristics
    @assert is_Unitary(mat,dim)
    # This function computes if there is any gate in the standard library 
    found_reduction = false
    for k in 2:dim
        push!(single_qubit_gates,R_x(pi/(2^k)))
        push!(single_qubit_gates,R_y(pi/(2^k)))
    end
    for single in single_qubit_gates
        for pos in 1:dim
            left_list = detect_reduction(mat * U_at(single,pos,dim),dim)
            right_list = detect_reduction(U_at(single,pos,dim) * mat,dim)
            if !isempty(left_list)
                println("Reduction of dimension detected with single left qubit gate!")
                found_reduction = true
                new_mat = trace_away(left_list, mat * U_at(single,pos,dim), dim)
                @assert (is_Unitary(new_mat,dim-length(left_list)) && is_Unitary(single',1))
                return [(new_mat, participating_qubits(left_list, list_qubit)),(single',[list_qubit[pos]])]
            elseif !isempty(right_list)
                println("Reduction of dimension detected with single right qubit gate!")
                found_reduction = true
                new_mat = trace_away(right_list,U_at(single,pos,dim) * mat , dim)
                @assert (is_Unitary(new_mat,dim-length(right_list)) && is_Unitary(single',1))
                return [(single',[list_qubit[pos]]),(new_mat, participating_qubits(right_list,list_qubit))]
            end
        end
    end
    for start in 1:dim
        for ind in start:dim
            for phi in [2*pi/(2^k) for k in 1:dim]
                left_list = detect_reduction(mat * get_U(phi,dim,start,ind),dim)
                right_list = detect_reduction(get_U(phi,dim,start,ind) * mat,dim)
                if  !isempty(left_list)
                    println("Reduction of dimension detected with Cphase gate!")
                    found_reduction = true
                    new_mat = trace_away(left_list,mat * get_U(phi,dim,start,ind), dim)
                    @assert (is_Unitary(get_U(phi,dim,1,2)',2) && is_Unitary(new_mat, dim-length(left_list)))
                    return [(new_mat,participating_qubits(left_list,list_qubit)),(get_U(phi,dim,1,2)',[list_qubit[start],list_qubit[ind]])]
                elseif !isempty(right_list)
                    println("Reduction of dimension detected with Cphase gate!")
                    found_reduction = true
                    new_mat = trace_away(right_list,get_U(phi,dim,start,ind) * mat, dim)
                    @assert (is_Unitary(get_U(phi,dim,1,2)',2) && is_Unitary(new_mat, dim-length(right_list)))
                    return [(get_U(phi,dim,1,2)',[list_qubit[start],list_qubit[ind]]),(new_mat,participating_qubits(right_list,list_qubit))]
                end
            end
        end
    end
    for double in two_qubit_gates
        for start in 1:dim-1
            for ind in start+1:dim
                left_list = detect_reduction(mat * kron(I(2^(start-1)),kron(skip_N(double,2,1,ind-start-1),I(2^(dim-ind)))),dim)
                right_list = detect_reduction(kron(I(2^(start-1)),kron(skip_N(double,2,1,ind-start-1),I(2^(dim-ind)))) * mat,dim)
                if !isempty(left_list)
                    println("Reduction of dimension detected with two qubit gate!")
                    found_reduction = true
                    new_mat = trace_away(left_list,mat * kron(I(2^(start-1)),kron(skip_N(double,2,1,ind-start-1),I(2^(dim-ind)))), dim)
                    @assert size(new_mat)[1] == 2^(dim-length(left_list))
                    @assert is_Unitary(double',2)
                    @assert is_Unitary(new_mat, dim-length(left_list))
                    return [(new_mat,participating_qubits(left_list,list_qubit)),(double',[list_qubit[start],list_qubit[ind]])]
                elseif !isempty(right_list)
                    println("Reduction of dimension detected with two qubit gate!")
                    found_reduction = true
                    new_mat = trace_away(right_list, kron(I(2^(start-1)),kron(skip_N(double,2,1,ind-start-1),I(2^(dim-ind)))) * mat, dim)
                    @assert (is_Unitary(double,2) && is_Unitary(new_mat, dim-length(right_list)))
                    return [(double',[list_qubit[start],list_qubit[ind]]),(new_mat,participating_qubits(left_list,list_qubit))]
                end
        
            end
        end
    end
    return []
end




function rotation(n, dim, l_mode)
    # First put the eigenvec and eigenvals pairs into degenerate eigenspace buckets
    # Then apply rotation matrix to rotate the subspaces respectively
    # Minimize linear independency (need to quantify)
    # assertion true that M^2 is part of 
    σzₙ = get_sigma_n(false,dim)
    σxₙ = get_sigma_n(true,dim)
    @assert is_Unitary(n,dim)
    if sum(abs.(n .- I(2^dim))) < error_threshold
        # Then the G to be decomposed is altogether in the k subalgebra
        return I(2^dim), I(2^dim), zeros(2^(dim-1))
    end 
    if l_mode
        println("Performing l decomposition...")
    else
        println("Performing m decomposition...")
    end
    if rounded
        decomp = schur(round.(n,digits=precision))
    else
        decomp = schur(n)
    end

    p̃ = decomp.vectors
    e = decomp.values
    if print_verbose
        println("Eigenvalues before processing: ", e)
    end
    @assert is_Unitary(p̃,dim)
    
    pos_count, neg_count,zeros_count = 0,0,0
    νⱼ, ν₊, ν₋ = [],[],[]
    K = Matrix{ComplexF64}(undef, 2^dim, 2^dim)
    e_comps = [] # To store all the distinct complex eigenvalues
    nums = []
    vec_comps = [] # To store all the complex eigenvectors
    vec_conj_comps = [] # And eigenvectors with e-vals that are complex conjugates with the ones above
    c_plus, c_minus = [],[]
    pos_plus,pos_minus,neg_plus, neg_minus= [],[],[],[],[]
    for k in 1:2^dim
        val, col = e[k], p̃[:,k]
        # For each val-vec pair, sort vecs into corresponding value buckets
        if abs(imag(val)) > error_threshold # If the eigenvalue is imaginary
            (is_in_e, ind_e) = is_in(val, e_comps,3)
            (is_in_e_hat, ind_e_hat) = is_in(conj(val), e_comps,3)
            if is_in_e # This means the eigenvalue is already registered
                nums[ind_e] += 1
                if sum(abs.(col .- σzₙ * col)) < error_threshold
                    push!(c_plus[ind_e], col)
                elseif sum(abs.(col .+ σzₙ * col)) < error_threshold
                    push!(c_minus[ind_e], col)
                else
                    push!(vec_comps[ind_e],col) # Push the eigenvector to the subvector corresponding to the eigenvalue
                end
            elseif is_in_e_hat # This means the e-val is conj of one of the registered distinct eigenvalues
                nums[ind_e_hat] += 1
                if sum(abs.(col .- σzₙ * col)) < error_threshold
                    push!(c_plus[ind_e_hat], col)
                elseif sum(abs.(col .+ σzₙ * col)) < error_threshold
                    push!(c_minus[ind_e_hat], col)
                else
                    push!(vec_conj_comps[ind_e_hat], col) # Push the vector into the corresponding conj vec list
                end
            else
                push!(e_comps, val) # Current e-val is not registered in the distinct eval list, push it to end
                if sum(abs.(col .- σzₙ * col)) < error_threshold
                    push!(c_plus, [col])
                    push!(vec_comps,[])
                    push!(c_minus,[])
                elseif sum(abs.(col .+ σzₙ * col)) < error_threshold
                    push!(c_minus, [col])
                    push!(vec_comps,[])
                    push!(c_plus,[])
                else
                    push!(vec_comps,[col]) # Initiate subvector e-vec
                    push!(c_minus,[])
                    push!(c_plus,[])
                end
                push!(vec_conj_comps,[]) # Initiate subvector for the conjugate e-vecs
                push!(nums,1)
            end
        elseif real(val) > 0.0  # If . is real psd
            pos_count += 1
            if (sum(abs.(col .- σzₙ * col)) < error_threshold) 
                push!(pos_plus, col)
            elseif (sum(abs.(col .+ σzₙ * col)) < error_threshold)
                push!(pos_minus, col)
            else
                push!(ν₊,col)
            end
        else
            neg_count += 1
            if (sum(abs.(col .- σzₙ * col)) < error_threshold)
                push!(neg_plus, col)
            elseif (sum(abs.(col .+ σzₙ * col)) < error_threshold)
                push!(neg_minus, col)
            else
                push!(ν₋, col)
            end          
        end
    end    
    if (sum(abs.(n .+ I(2 ^ dim))) < error_threshold) || (sum(abs.(n .+ im * I(2 ^ dim))) < error_threshold) || (sum(abs.(n .- im * I(2 ^ dim))) < error_threshold)
        K = I(2 ^ dim)
    elseif !((length(e_comps)+length(nums)+length(vec_comps)+length(c_plus)+length(c_minus)+length(vec_conj_comps)) == 6  * length(e_comps))
        println("length of e_comps: ", length(e_comps))
        println("length of nums: ", length(nums))
        println("length of vec_comps: ", length(vec_comps))
        println("length of vec_conj_comps: ", length(vec_conj_comps))
        println("length of c_minus:", length(c_minus))
        println("length of c_plus:", length(c_plus))
    end
    if print_verbose
        println("number of pos vec: ", pos_count,length(ν₊))
        println("number of neg vec: ", neg_count,length(ν₋))
        println("number of comp vec: ", length(nums) > 0 ? sum(nums) : 0)
        println("List of complex eigenvalue sorted by algorithm: ", e_comps)
        println("Corresponding numbers for each complex eigenvalue: ", nums)
        println("length of singular pos, neg vectors: ", length(pos_plus),length(pos_minus), length(neg_plus), length(neg_minus))
    end


    counter = 0
    
    for (total_count, normal_list, list_conj, plus, minus) in [[(pos_count,ν₊,[],pos_plus,pos_minus),(neg_count,ν₋,[], neg_plus, neg_minus)];[(nums[i],vec_comps[i],vec_conj_comps[i],c_plus[i],c_minus[i]) for i in 1:length(e_comps)]]
        if total_count > 0
            pair_length = minimum([length(plus),length(minus)])
            if print_verbose
                println("Total, numbers ", total_count, length(normal_list), length(list_conj), length(plus), length(minus))
                println("Total number of eigenvectors correspond to the eigenvalue: ", total_count)
                println("Pair length: ", pair_length)
                println("Total length of available vectors that are not singular: ", (length(normal_list) + length(list_conj)))
                println("Required number of basis vectors: ", total_count/2 - pair_length)
                println("Number of plus singular vectors: ", length(plus))
                println("Number of minus singular vectors: ", length(minus))
            end
            used_pairs = (pair_length > 0) && (length(normal_list) + length(list_conj)) < total_count/2 
            pairs_list = []
            if used_pairs
                # Only if the normal list is not enough to  enough basis involve the singular ones
                for i in 1: pair_length
                    if l_mode
                        println("Singular plus vectors used as basis for l decomposition.")
                        K[:,counter + 2*i-1] = normalize(plus[i])
                        K[:,counter + 2*i] = normalize(σxₙ * K[:,counter + 2*i-1])
                        pairs_list = plus[1:pair_length]
                    else
                        println("Singular plus and minus vectors used as basis for m decomposition.")
                        K[:,counter + 2*i-1] = normalize(plus[i])
                        K[:,counter + 2*i] = normalize(minus[i])
                        pairs_list = [plus[1:pair_length];minus[1:pair_length]]
                    end
                end
                counter += 2 * pair_length
            end
            if isempty(list_conj)
                if used_pairs
                    is_found, basis = construct_basis(normal_list, Int(total_count/2), σzₙ, dim, pairs_list)
                else
                    is_found, basis = construct_basis(normal_list, Int(total_count/2 - pair_length), σzₙ, dim, [])
                end
            elseif l_mode
                if used_pairs
                    is_found, basis = construct_basis([normal_list;list_conj], Int(total_count/2), σzₙ, dim, pairs_list)
                else
                    is_found, basis = construct_basis([normal_list;list_conj], Int(total_count/2 - pair_length), σzₙ, dim, [])
                end
            else
                println("Choose complex basis vectors and they should technically be orthogonal to each other after manipulation...")
                if (used_pairs && length(normal_list) >= total_count/2 - pair_length) || (!used_pairs && (length(normal_list) >= total_count/2))
                    is_found, basis = true, normal_list
                elseif (used_pairs && length(list_conj) >= total_count/2 - pair_length) || (!used_pairs && (length(list_conj) >= total_count/2))
                    is_found, basis = true, list_conj
                else
                    println("Too bad now you have to code it.")
                    is_found, basis = construct_complex_basis(normal_list, list_conj, total_count/2 - pair_length, σzₙ, dim)
                end
            end
            if is_found 
                println("Valid solution is found!")
                for i in 1:Int(total_count/2 - pair_length)
                    if l_mode
                        K[:,counter + 2*i-1] = normalize(basis[i] .+ σzₙ * basis[i])
                        K[:,counter + 2*i] = normalize(σxₙ * K[:,counter + 2*i-1])
                    else
                        K[:,counter + 2*i-1] = normalize(basis[i] .+ σzₙ * basis[i])
                        K[:,counter + 2*i] = normalize(basis[i] .- σzₙ * basis[i])
                    end
                end     
                counter += 2 * Int(total_count/2 - pair_length)
            elseif total_count > 2 * pair_length
                throw(error("No suitable basis found, this is very bad."))
            end  
        end
    end
    if print_verbose
        println([round(K[:,i]' * K[:,i],digits=2) for i in 1:2^dim])
    end
    if !is_Unitary(K,dim)
        println("Difference from unitary: ", sum(abs.(K' * K .- I(2^dim))))
        display(round.(K' * K,digits=2))
        display(round.(K,digits=2))
    end
    @assert is_Unitary(K,dim)
    @assert sum(abs.(σzₙ * K * σzₙ .- K)) < error_threshold
    if l_mode
        @assert sum(abs.(σxₙ * K * σxₙ .- K)) < error_threshold
    end
    b = K' * round.(n,digits=precision) * K
    # b = h^2, which is the cartan component that corresponds to h = p(dagger)mp, hence m = pmp(dagger), i.e. the rotation we need
    ζ_store = zeros(ComplexF64, 2^(dim-1))
    if l_mode
        op = sz
    else
        op = sx
    end
    for i in 1: 2^(dim-1)
        proj = one_at(i,2^(dim-1))
        right = kron(proj, I(2))
        left = kron(proj', I(2))
        R = left * b * right
        # Now Rx is a 2 * 2 matrix
        if sum(abs.(R .- exp(im * pi *op))) < error_threshold
                ζ = π
        else 
            ζ =  real((get_log(R) * op)[1,1]/im)
        end

        ζ_store[i] = ζ
        if sum(abs.(R .- exp(im * ζ * op))) > error_threshold
            println("error while reconstructing: ")
            display(R)
            println("zeta recoreded: ", ζ)
        end
        @assert sum(abs.(R .- exp(im * ζ * op))) < error_threshold
    end
    b_reconstructed = reduce(+, [kron((one_at(i,2^(dim-1)) * one_at(i,2^(dim-1))'),exp(im * op * ζ_store[i])) for i in 1:2^(dim-1)])
    if sum(abs.(b * b' .- I(2^dim))) > error_threshold
        println("difference from unitary: ",sum(abs.(b * b' .- I(2^dim))))
    end
    @assert is_Unitary(b,dim)
    if sum(abs.(b .- b_reconstructed)) > error_threshold
        println("Error of reconstruction: ",sum(abs.(b .- b_reconstructed)))
    end
    @assert sum(abs.(b .- b_reconstructed)) < error_threshold
    y = reduce(+, [kron((one_at(i,2^(dim-1)) * one_at(i,2^(dim-1))') , exp(im * op * ζ_store[i]/2)) for i in 1:2^(dim-1)])
    @assert is_Unitary(y,dim)
    # Here M is the lie group element
    M = K * y * K'
    @assert is_Unitary(M,dim)
    if l_mode
        @assert sum(abs.(σxₙ * M * σxₙ .- M')) < error_threshold # Now the M element anticommutes with sx
        @assert sum(abs.(σzₙ * M * σzₙ .- M)) < error_threshold # In L decomposition all matrices already satisfy commutation relation
    else
        @assert sum(abs.(σzₙ * M * σzₙ .- M')) < error_threshold
    end
    # From zeta_store/2 one can reconstruct the h element by sum(|j><j| kron Rx(zeta/2))
    return K, M, ζ_store/2
end
