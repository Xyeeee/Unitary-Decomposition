using Plots
using JSON
include("Utils.jl")
 
#*=========Below is the configuration block for Decomposition.jl================*#
# Number of nuclear physical qubits involved = num_qubit = dim
dim = 2
# Definition of target gate
target_gate = get_SWAP()


# Location of the dump file for decomposed sequence
file_loc = "test.json"
#*==============================================================================*#

function get_decomposed_sequence(target_gate, dim, list_qubit)
    @assert is_Unitary(target_gate,dim)
    if round_input
        target_gate = round.(target_gate, digits=input_precision)
    end
    if 1-abs(tr(I(2^dim) * target_gate) / tr(I(2^dim)' * I(2^dim))) < error_threshold
        return []
    end
    if dim == 1
        @assert is_Unitary(target_gate,dim)
        g = get_log(target_gate)/im
        coef =  real.(tr.([g * sx, g* sy, g*sz]))/2
        @assert (1-abs(tr(target_gate' * exp(im * sum(coef .* [sx,sy,sz]))) / tr(target_gate' * target_gate))) < error_threshold
        if sum(abs.(coef)) < error_threshold
            return []
        else
            return [Dict("type"=>"N", "qubit_list"=> [last(list_qubit)], "coef" => coef)]
        end
    elseif dim==2 && use_canonical_su4
        local_seq = []
        g = get_log(target_gate)
        coef, g_m = get_coef_2(g,true)
        m = solve_m(g,g_m)
        K = target_gate * exp(-m)
        K_1 = trace_at(K, dim, 2)
        K_2 = trace_at(K, dim, 1)
        m_XX = 1/4 * tr(m * kron(sx,sx)) * im
        m_YY = 1/4 * tr(m * kron(sy,sy)) * im
        m_ZZ = 1/4 * tr(m * kron(sz,sz)) * im
        append!(local_seq,[get_decomposed_sequence(K_1,1,list_qubit[1]);get_decomposed_sequence(K_2,1,list_qubit[2])])
        append!(local_seq,[Dict("type"=>"N", "no_q"=> dim, "qubit_list"=> [list_qubit[2]],"coef" => [0.0, -pi/4, 0.0]), 
                          Dict("type"=>"N", "no_q"=> dim, "qubit_list"=> [list_qubit[1]],"coef" => [0.0, -pi/4, 0.0]), 
                          Dict("type"=>"NV", "dim"=>dim,"qubit_list"=> list_qubit, "coef"=> real.(m_XX) * ZZ_coef), 
                          Dict("type"=>"N", "no_q"=> dim, "qubit_list"=>[list_qubit[2]],"coef" => [0.0, pi/4, 0.0]),
                          Dict("type"=>"N", "no_q"=> dim, "qubit_list"=>[list_qubit[1]],"coef" => [0.0, pi/4, 0.0])])
        append!(local_seq,[Dict("type"=>"N", "no_q"=> dim, "qubit_list"=> [list_qubit[2]],"coef" => [pi/4, 0.0, 0.0]), 
                          Dict("type"=>"N", "no_q"=> dim, "qubit_list"=> [list_qubit[1]],"coef" => [pi/4, 0.0, 0.0]), 
                          Dict("type"=>"NV", "dim"=>dim,"qubit_list"=> list_qubit, "coef"=> real.(m_YY) * ZZ_coef), 
                          Dict("type"=>"N", "no_q"=> dim, "qubit_list"=>[list_qubit[2]],"coef" => [-pi/4, 0.0, 0.0]),
                          Dict("type"=>"N", "no_q"=> dim, "qubit_list"=>[list_qubit[1]],"coef" => [-pi/4, 0.0, 0.0])])
        append!(local_seq,[Dict("type"=>"NV", "dim"=>dim,"qubit_list"=> list_qubit, "coef"=> real.(m_ZZ) * ZZ_coef)])
        return local_seq
    else
        local_seq = []
        prod_list = separation(target_gate,list_qubit,dim)
        if length(prod_list) > 1
            println(prod_list)
            for (gate,parlist,pardim) in prod_list
                append!(local_seq, get_decomposed_sequence(gate,pardim, parlist))
            end
            @assert sum(abs.(get_total_unitary(target_gate,num_qubit,list_qubit) .- unit)) < error_threshold
            println("from separation returning local sequence: ", local_seq)
            return local_seq
        elseif dim > 2 && !isempty(dim_reduction(target_gate,dim,list_qubit))
            red_list = dim_reduction(target_gate,dim,list_qubit)
            println("Now we process the reduced sequence!")
            println("length of list: ", length(red_list))
            for (mat,par_list) in red_list
                append!(local_seq,get_decomposed_sequence(mat,length(par_list),par_list))
            end
            return local_seq
        end


        σzₙ = get_sigma_n(false,dim)
        σxₙ = get_sigma_n(true,dim)
        @assert is_Unitary(target_gate,dim)

        short_list = list_qubit[1:length(list_qubit)-1]

        g = get_log(target_gate)

        # If the central element is diagonal, immediately return, no need to further decompose, we're done, return the cached local sequence.
        if sum(abs.(σzₙ * target_gate * σzₙ .- target_gate)) < error_threshold 
            println("No need to m decompose here!")
            K_0 = target_gate
            K_1 = I(2^dim)
            H_central = zeros(2^dim)
        elseif is_diagonal(g) 
            println("Gate to be decomposed is DIAGONAL!")
            println("Decomposition at dim = ", dim, " has central diagonal coefs:")
            diag_coef = [g[i,i] for i in 1:2^dim] 
            println(diag_coef)
            if sum(abs.(diag_coef)) < error_threshold
                return []
            else
                return [Dict("type"=>"NV", "dim"=>dim, "coef"=> imag.(diag_coef))]
            end
        else
            K_1, M, H_list_central = rotation(get_m_squared(target_gate, false, dim),dim,false)
            # Checking Unitary condition is met with output matrix K_1
            @assert is_Unitary(K_1,dim)
            K_0 = target_gate * M'
            @assert is_Unitary(K_0,dim)
            if sum(abs.(σzₙ * K_0 * σzₙ .- K_0)) >  error_threshold
                println("Error of commutivity: ",sum(abs.(σzₙ * K_0 * σzₙ .- K_0)))
            end
            @assert sum(abs.(σzₙ * K_0 * σzₙ .- K_0)) < error_threshold 
            println("Decomposition at dim = ", dim, " has central diagonal coefs:")
            println(round.(H_list_central,digits=5))
            H_central = [ 
                is_conj ? -i : i 
                for i in H_list_central 
                for is_conj in [false; true]
            ]
            H = kron(I(2^(dim-1)),exp(im * -pi/4 * sy)) * exp(im * diagm(H_central)) * kron(I(2^(dim-1)),exp(im * pi/4 * sy))
            @assert sum(abs.(K_0 * K_1 * H * K_1' .- target_gate)) < error_threshold 
        end

        # log(K_0 K_1) resides in l subspace, spanning I + su(2^dim-1) x z,  su(2^dim - 1)
        g = get_log(K_0 * K_1)
        @assert sum(abs.(σzₙ * K_0 * K_1 * σzₙ .- K_0 * K_1)) < error_threshold
        if  sum(abs.(σxₙ * K_0 * K_1 * σxₙ .- K_0 * K_1)) < error_threshold
            println("No need of l decomposition here!")
            # Then the gate to be l decomposed already belongs to the SU(2^(n-1)) group
            append!(local_seq, get_decomposed_sequence(trace_over(K_0 * K_1,dim,dim-1),dim-1, short_list))
            if sum(abs.(H_central)) > error_threshold
                append!(local_seq, [Dict("type"=>"N", "no_q"=> dim, "qubit_list"=> [last(list_qubit)],"coef" => [0.0, -pi/4, 0.0]),Dict("type"=>"NV", "dim"=>dim,"qubit_list"=> list_qubit, "coef"=> real.(H_central)),Dict("type"=>"N", "no_q"=> dim, "qubit_list"=>[last(list_qubit)],"coef" => [0.0, pi/4, 0.0])])
            end
        elseif is_diagonal(g)
            println("Gate to be decomposed is DIAGONAL!")
            println("Decomposition at dim = ", dim, " has left l subalgebra diagonal coefs:")
            diag_coef_l = [g[i,i] for i in 1:2^dim]
            println(diag_coef_l)
            K_0_left = I(2^dim)
            K_1_left = I(2^dim)
            if sum(abs.(diag_coef_l)) > error_threshold && sum(abs.(H_central)) > error_threshold
                append!(local_seq, [Dict("type"=>"NV", "dim"=>dim, "qubit_list"=> list_qubit,"coef"=> imag.(diag_coef_l)),Dict("type"=>"N", "no_q"=> dim, "qubit_list"=> [last(list_qubit)],"coef" => [0.0, -pi/4, 0.0]),Dict("type"=>"NV", "dim"=>dim,"qubit_list"=> list_qubit, "coef"=> real.(H_central)),Dict("type"=>"N", "no_q"=> dim, "qubit_list"=>[last(list_qubit)],"coef" => [0.0, pi/4, 0.0])])
            elseif sum(abs.(diag_coef_l)) > error_threshold
                append!(local_seq, [Dict("type"=>"NV", "dim"=>dim, "qubit_list"=> list_qubit,"coef"=> imag.(diag_coef_l))])
            elseif sum(abs.(H_central)) > error_threshold
                append!(local_seq, [Dict("type"=>"N", "no_q"=> dim,"qubit_list"=> [last(list_qubit)], "coef" => [0.0, -pi/4, 0.0]),Dict("type"=>"NV", "dim"=>dim,"qubit_list"=> list_qubit, "coef"=> real.(H_central)),Dict("type"=>"N", "no_q"=> dim, "qubit_list"=> [last(list_qubit)],"coef" => [0.0, pi/4, 0.0])])
            end
        else
            K_1_left, M, H_list_l = rotation(get_m_squared(K_0*K_1, true, dim),dim,true)
            K_0_left = K_0 * K_1 * M'
            @assert is_Unitary(K_1_left,dim) && is_Unitary(K_0_left,dim)
            @assert sum(abs.(σxₙ * K_0_left * σxₙ .- K_0_left)) < error_threshold 
            @assert sum(abs.(σxₙ * K_0_left' * σxₙ .- K_0_left')) < error_threshold
            @assert sum(abs.(σzₙ * K_0_left * K_1_left * σzₙ .- K_0_left* K_1_left)) < error_threshold
            @assert sum(abs.(σzₙ * K_1_left' * σzₙ .- K_1_left')) < error_threshold
            @assert sum(abs.(σxₙ * K_1_left' * σxₙ .- K_1_left')) < error_threshold
            println("Decomposition at dim = ", dim, " has left l subalgebra diagonal coefs:")
            println(round.(H_list_l,digits=5))
            H_l = [ 
                is_conj ? -i : i 
                for i in H_list_l
                for is_conj in [false; true]
            ]
            @assert sum(abs.(K_0_left * K_1_left * exp(im * diagm(H_l)) * K_1_left' .- K_0 * K_1)) < error_threshold
            append!(local_seq, get_decomposed_sequence(trace_over(K_0_left * K_1_left,dim,dim-1),dim-1, short_list))
            @assert sum(abs.(kron(trace_over(K_0_left * K_1_left,dim,dim-1), I(2)) .- K_0_left * K_1_left)) < error_threshold
            if sum(abs.(H_l)) > error_threshold
                append!(local_seq, [Dict("type"=>"NV", "dim"=>dim, "qubit_list"=> list_qubit,"coef"=> real.(H_l))])
            end
            append!(local_seq, get_decomposed_sequence(trace_over(K_1_left',dim,dim-1),dim-1,short_list))
            let err = sum(abs.(kron(trace_over(K_1_left',dim,dim-1), I(2)) .- K_1_left'))
                if err > error_threshold
                    println(err)
                    display(round.(K_1_left',digits=2))
                    println(sum(abs.(kron(trace_over(K_1_left',dim,dim-1), I(2)) .- K_1_left')))
                    @assert err < error_threshold
                end
            end
            if sum(abs.(H_central)) > 1e-5
                append!(local_seq, [Dict("type"=>"N", "no_q"=> dim,"qubit_list"=> [last(list_qubit)], "coef" => [0.0, -pi/4, 0.0]),Dict("type"=>"NV", "dim"=>dim,"qubit_list"=> list_qubit, "coef"=> real.(H_central)),Dict("type"=>"N", "no_q"=> dim,"qubit_list"=> [last(list_qubit)],"coef" => [0.0, pi/4, 0.0])])
            end
        end

        # Right part is K_1 daggertrace_10 = I(2^(dim-1)) ⊗ [1, 0]
        g = get_log(K_1')
        if  sum(abs.(σxₙ * K_1' * σxₙ .- K_1')) < error_threshold
            println("No need of l decomposition here!")
            # Then the gate to be l decomposed already belongs to the SU(2^(n-1)) group
            append!(local_seq, get_decomposed_sequence(trace_over(K_1',dim,dim-1),dim-1, short_list))
        elseif is_diagonal(g)
            println("Gate to be decomposed is DIAGONAL!")
            println("Decomposition at dim = ", dim, " has right l subalgebra diagonal coefs:")
            diag_coef_r = [g[i,i] for i in 1:2^dim]
            println(diag_coef_r)
            K_0_right = I(2^dim)
            K_1_right = I(2^dim)
            if sum(abs.(diag_coef_r)) > error_threshold
                append!(local_seq, [Dict("type"=>"NV", "dim"=>dim, "qubit_list"=> list_qubit,"coef"=> imag.(diag_coef_r))])
            end
        else
            K_1_right, M, H_list_r = rotation(get_m_squared(K_1', true, dim),dim,true)
            K_0_right = K_1' * M'
            @assert is_Unitary(K_1_right,dim) && is_Unitary(K_0_right,dim)
            @assert sum(abs.(σxₙ * K_0_right * σxₙ .- K_0_right)) < error_threshold
            @assert sum(abs.(σzₙ * K_0_right' * σzₙ .- K_0_right')) < error_threshold
            @assert sum(abs.(σzₙ * K_0_right * K_1_right * σzₙ .- K_0_right * K_1_right)) < error_threshold
            @assert sum(abs.(σxₙ * K_0_right * K_1_right * σxₙ .- K_0_right * K_1_right)) < error_threshold
            @assert sum(abs.(σzₙ * K_1_right' * σzₙ .- K_1_right')) < error_threshold
            @assert sum(abs.(σxₙ * K_1_right' * σxₙ .- K_1_right')) < error_threshold
            println("Decomposition at dim = ", dim, " has right l subalgebra diagonal coefs:")
            println(round.(H_list_r,digits=5))
            H_r = [ 
                is_conj ? -i : i 
                for i in H_list_r
                for is_conj in [false; true]
            ]
            tar = K_0_right * K_1_right
            @assert sum(abs.(K_0_right * K_1_right * exp(im * diagm(H_r)) * K_1_right' .- K_1')) < error_threshold
            @assert sum(abs.(kron(trace_over(tar,dim,dim-1), I(2)) .- tar)) < error_threshold
            append!(local_seq, get_decomposed_sequence(trace_over(K_0_right * K_1_right,dim,dim-1),dim-1,short_list))
            if sum(abs.(H_r)) > error_threshold
                append!(local_seq, [Dict("type"=>"NV", "dim"=>dim, "qubit_list"=> list_qubit,"coef"=> real.(H_r))])
            end
            append!(local_seq, get_decomposed_sequence(trace_over(K_1_right',dim,dim-1),dim-1,short_list))
            @assert sum(abs.(kron(trace_over(K_1_right',dim,dim-1), I(2)) .- K_1_right')) < error_threshold
        end

    return local_seq
    end
end


function compile_gate(target_gate, dim, file_loc)
    seq = get_decomposed_sequence(target_gate,dim,1:1:dim)
    gate_dict = Dict("seq"=>seq)
    open(file_loc,"w") do f
        JSON.print(f,gate_dict) 
    end
    return length(seq)
end

compile_gate(target_gate, dim, file_loc)
