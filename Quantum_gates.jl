using LinearAlgebra


⊗(a,b) = kron(a,b)


# Utility functions
function one_at(ind, len)
    arr = zeros(len)
    arr[ind] = 1
    return arr
end

function R_x(theta)
    return exp(im * theta * sx)
end

function R_y(theta)
    return exp(im * theta * sy)
end

# Returns X rotation operator on ind th nuclear spin
function X_at(ind, num_qubit)
    return kron(kron(I(2^(ind-1)), sx), I(2^(num_qubit-ind))) 
end
# Returns Y rotation operator on ind th nuclear spin
function Y_at(ind, num_qubit)
    return kron(kron(I(2^(ind-1)), sy), I(2^(num_qubit-ind))) 
end
# Returns Z rotation operator on ind th nuclear spin
function Z_at(ind, num_qubit)
    return kron(kron(I(2^(ind-1)), sz), I(2^(num_qubit-ind))) 
end

function U_at(unitary, ind, num_qubit)
    return kron(I(2^(ind-1)),kron(unitary, I(2^(num_qubit-ind))))
end

#--------------Constant matrix definitions------------------#
function get_SWAP()
    SWAP = zeros(ComplexF64, 4, 4)
    SWAP[1,1] = 1
    SWAP[2,3] = 1
    SWAP[3,2] = 1
    SWAP[4,4] = 1
    return SWAP
end

function get_C_Z()
    C_Z = diagm([1,1,1,1])
    C_Z[4,4] = -1
    return C_Z
end


function get_C_NOT()
    C_NOT = zeros(ComplexF64, 4, 4)

    C_NOT[1,1] = 1
    C_NOT[2,2] = 1
    C_NOT[3,4] = 1
    C_NOT[4,3] = 1
    return C_NOT
end

function get_H()
    Hadamard = diagm([1.0,-1.0])
    Hadamard[1,2] = 1.0
    Hadamard[2,1] = 1.0
    return Hadamard /sqrt(2)
end



function get_U(phi, dim, controll_ind, target_ind)
    # When it is the control ind: state projection for the control bit
    # When it is the target, the 1 state gets the phase gate and 0 state gets identity
    # Otherwise it's identity with both
    U_single =  diagm([1.0, exp(im * phi)])
    zero = 1.0
    one = 1.0
    for ind in 1:dim
        if ind == controll_ind
            zero = kron(zero,p_0)
            one = kron(one, p_1)
        elseif ind == target_ind
            zero = kron(zero,I(2))
            one = kron(one, U_single)
        else
            zero = kron(zero, I(2))
            one = kron(one ,I(2))
        end
    end
    return zero .+ one
end

function get_control_not(dim, controll_ind, target_ind)
    # When it is the control ind: state projection for the control bit
    # When it is the target, the 1 state gets the phase gate and 0 state gets identity
    # Otherwise it's identity with both
    U_single =  sx
    zero = 1.0
    one = 1.0
    for ind in 1:dim
        if ind == controll_ind
            zero = kron(zero,p_0)
            one = kron(one, p_1)
        elseif ind == target_ind
            zero = kron(zero,I(2))
            one = kron(one, U_single)
        else
            zero = kron(zero, I(2))
            one = kron(one ,I(2))
        end
    end
    return zero .+ one
end

function get_QFT(N)
    # According to the wikipedia definition
    mat = I(2^N)
    for i in 1:N
        mat = U_at(get_H(),i,N) * mat
        # println("Adding H to qubit ", i)
        @assert is_Unitary(mat,N)
        for j in i+1:N
            mat = get_U(2 * pi/(2^(j-i+1)),N,j,i) * mat
            # println("Adding Cphase with control ", j, " and target ",i)
            @assert is_Unitary(mat,N) 
        end
    end
    return mat
end


function C_U_map(control_bin, target_bin, is_active)
    if is_active 
        if control_bin == 1
            return p_1
        elseif control_bin == -1
            return p_0
        end
        if target_bin == 1
            return sx
        elseif target_bin == -1
            return sz
        end
        return I(2)
    else
        if control_bin == 1
            return p_1
        elseif control_bin == -1
            return p_0
        end
        return I(2)
    end
end


function controlled_gates(controls, targets, dim)
    # Controlls is a list of binaries over the set of qubits, where 1 label solid dot and -1 label empty dot
    # Targets is the list of binaries over the set of qubits, where 1 label X and -1 label z
    active = reduce(⊗, [C_U_map(controls[i], targets[i],true) for i in 1:dim])
    inactive = reduce(⊗, [C_U_map(controls[i], targets[i],false) for i in 1:dim])
    return Matrix{ComplexF64}(I(2^dim) .+ active .- inactive)
end

function get_CNOTNOT()
    alternative = controlled_gates([1,0,0],[0,1,1],3)
    return alternative
end

function get_Toffoli()
    mat = diagm([ones(6);zeros(2)])
    mat[7,8] = 1.0
    mat[8,7] = 1.0
    return Matrix{ComplexF64}(mat)
end



function get_PERM()
    thingy = kron(get_SWAP(), I(2)) * skip_N(get_SWAP(),2,1,1)
    return Matrix{ComplexF64}(thingy)
end

function get_ecc_regina()
    return controlled_gates([0,0,0,1,-1],[1,0,0,0,0],5)*controlled_gates([0,0,0,1,1],[0,1,0,0,0],5) * controlled_gates([0,0,0,1,1],[0,0,1,0,0],5) * controlled_gates([0,1,0,0,0],[0,0,0,1,1],5) * controlled_gates([1,0,0,0,0],[0,0,0,1,0],5)
end

function get_test_gate(r)
    return Matrix{ComplexF64}(reduce(+, [exp(im * 2 * pi * U_at(sz,i,r)* (i-1)/r) for i in 1:r])./sqrt(r))
end

function get_Ising(len, a, b, T)
    # Arguments
    # len: length of the spin chain
    # a: individual bias 
    # b: b_ij is the strength of coupling between neighboring spins
    H = a[1]* kron(sx ,I(2^(len-1)))
    for i in 2:len
        H += a[i] * kron(kron(I(2^(i-1)) , sx), I(2^(len-i)))
        H += b[i] * kron(kron(kron(I(2^(i-2)), sz), sz),I(2^(len-i)))
    end
    U = exp(im * H * T)
    return U
end

function get_example_uni()
    U = kron(diagm(one_at(1,4)),I(2)) + kron(diagm([0,1,1,0]), exp(im * pi/4 * sx)) + kron(diagm(one_at(4,4)), exp(im * pi/2 * sx))
    return U
end

function get_Grover()
    # This implementation involves 4 qubits, Grover refers to the Grover's search algorithm
    return reduce(*,
        [
            reduce(⊗, [get_H() * sx for i in 1:4]),
            get_U(pi/4, 4, 1, 4),
            get_C_NOT() ⊗ I(4),
            get_U(-pi/4,4, 2,4), 
            get_C_NOT() ⊗ I(4), 
            get_U(pi/4,4, 2,4), 
            I(2) ⊗ get_C_NOT() ⊗ I(2), 
            get_U(-pi/4, 4,3,4), 
            skip_N(get_C_NOT(),2,1,1) ⊗ I(2), 
            get_U(pi/4, 4,3,4), 
            I(2) ⊗ get_C_NOT() ⊗ I(2), 
            I(2) ⊗ sx ⊗ get_U(-pi/4,2,1,2), 
            skip_N(get_C_NOT(),2,1,1) ⊗ I(2), 
            (sx* get_H() * sx) ⊗ (get_H() * sx) ⊗ get_U(pi/4,2,1,2), 
            I(4) ⊗ (sx * get_H() * sx) ⊗ (sx * get_H() * sx), 
            get_U(pi/4, 4, 1,4)
        ]
    )
end

function G(a, b, c, d)
    M = zeros(4,4) + I(4)
    M[0b01+1, 0b01+1] = a
    M[0b01+1, 0b10+1] = b
    M[0b10+1, 0b01+1] = c
    M[0b10+1, 0b10+1] = d
    M
end

function G(theta)
    a = cos(theta)
    b = -sin(theta)
    c = sin(theta)
    d = cos(theta)
    G(a, b, c, d)
end

function G2(a, b, c, d)
    M = zeros(16, 16) + I(16)
    # from ground state (0011)
    M[0b0011+1, 0b0011+1] = a
    M[0b0011+1, 0b1100+1] = b
    M[0b1100+1, 0b0011+1] = c
    M[0b1100+1, 0b1100+1] = d

    # from first excited state (0101)
    M[0b0101+1, 0b0101+1] = a
    M[0b0101+1, 0b1010+1] = b
    M[0b1010+1, 0b0101+1] = c
    M[0b1010+1, 0b1010+1] = d

    # from second excited state (0110)
    M[0b0110+1, 0b0110+1] = a
    M[0b0110+1, 0b1001+1] = b
    M[0b1001+1, 0b0110+1] = c
    M[0b1001+1, 0b1001+1] = d

    M
end

function G2(theta)
    a = cos(theta)
    b = -sin(theta)
    c = sin(theta)
    d = cos(theta)
    return Matrix{ComplexF64}(G2(a, b, c, d))
end

function make_random_circuit(num_cnots, total_qubits)
    to_return = I(2^total_qubits)
    for i in 1:num_cnots
        c_ind = rand(1:total_qubits)
        t_ind = rand(1:total_qubits)
        while t_ind == c_ind
            t_ind = rand(1:total_qubits)
        end
        to_return = get_control_not(total_qubits, c_ind,t_ind) * to_return
    end
    return  Matrix{ComplexF64}(to_return)
end
