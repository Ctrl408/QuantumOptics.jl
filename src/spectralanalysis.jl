using Arpack
import KrylovKit: eigsolve
import SciMLBase
import ForwardDiff
using SparseArrays, LinearAlgebra

const nonhermitian_warning = "The given operator is not hermitian. If this is due to a numerical error make the operator hermitian first by calculating (x+dagger(x))/2 first."

"""
    abstract type DiagStrategy

Represents an algorithm used to find eigenvalues and eigenvectors of some operator.
Subtypes of this abstract type correspond to concrete routines. See `LapackDiag`,
`KrylovDiag` for more info.
"""
abstract type DiagStrategy end

"""
    LapackDiag <: DiagStrategy

Represents the `LinearAlgebra.eigen` diagonalization routine.
The only parameter `n` represents the number of (lowest) eigenvectors.
"""
struct LapackDiag <: DiagStrategy
    n::Int
end

"""
    KrylovDiag <: DiagStrategy

Represents the `KrylovKit.eigsolve` routine. Implements the Lanczos & Arnoldi algorithms.
"""
struct KrylovDiag{VT} <: DiagStrategy
    n::Int
    v0::VT
    krylovdim::Int
end

"""
    KrylovDiag(n::Int, [v0=nothing, krylovdim::Int=n + 30])

Parameters:
- `n`: The number of eigenvectors to find
- `v0`: The starting vector. By default it is `nothing`, which means it will be a random dense
`Vector`.
- `krylovdim`: The upper bound for dimension count of the emerging Krylov space.
"""
KrylovDiag(n::Int, v0=nothing) = KrylovDiag(n, v0, n + 30)
Base.print(io::IO, kds::KrylovDiag) =
    print(io, "KrylovDiag($(kds.n))")

arithmetic_unary_error = QuantumOpticsBase.arithmetic_unary_error

"""
    detect_diagstrategy(op::Operator; kw...)

Find a `DiagStrategy` for the given operator; processes the `kw` keyword arguments
and automatically sets parameters of the resulting `DiagStrategy` object.
Returns a tuple of the `DiagStrategy` and unprocessed keyword arguments from `kw`.
"""
function detect_diagstrategy(op::DataOperator; kw...)
    QuantumOpticsBase.check_samebases(op)
    detect_diagstrategy(op.data; kw...)
end
detect_diagstrategy(op::AbstractOperator; kw...) = arithmetic_unary_error("detect_diagstrategy", op)

"""
    get_starting_vector(m::AbstractMatrix)

Generate a default starting vector for Arnoldi-like iterative methods for matrix `m`.
"""
get_starting_vector(::SparseMatrixCSC) = nothing

function detect_diagstrategy(m::AbstractSparseMatrix; kw...)
    if get(kw, :info, true)
        @info "Defaulting to sparse diagonalization for sparse operator.
If storing the full operator is possible, it might be faster to do `eigenstates(dense(op))`.
Set `info=false` to turn off this message."
    end
    nev = get(kw, :n, 6)
    v0 = get(kw, :v0, get_starting_vector(m))
    krylovdim = get(kw, :krylovdim, nev + 30)
    new_kw = Base.structdiff(values(kw), NamedTuple{(:n, :v0, :krylovdim, :info)})
    return KrylovDiag(nev, v0, krylovdim), new_kw
end

function detect_diagstrategy(m::Matrix; kw...)
    nev = get(kw, :n, size(m)[1])
    new_kw = Base.structdiff(values(kw), NamedTuple{(:n, :info)})
    return LapackDiag(nev), new_kw
end

detect_diagstrategy(m::T; _...) where T<:AbstractMatrix = throw(ArgumentError(
    """Cannot detect DiagStrategy for array type $(typeof(m)).
    Consider defining `QuantumOptics.detect_diagstrategy(::$T; kw...)` method.
    Refer to `QuantumOptics.detect_diagstrategy` docstring for more info."""))

"""
    eigenstates(op::Operator[, n::Int; warning=true, kw...])

Calculate the lowest n eigenvalues and their corresponding eigenstates.
"""
function eigenstates(op::AbstractOperator; kw...)
    ds, kwargs_rem = detect_diagstrategy(op; kw...)
    eigenstates(op, ds; kwargs_rem...)
end

eigenstates(op::AbstractOperator, n::Int; warning=true, kw...) =
    eigenstates(op; warning=warning, kw..., n=n)

function eigenstates(op::Operator, ds::LapackDiag; warning=true)
    b = basis(op)
    data = op.data
    if ishermitian(op)
        # AD Fix: Use full eigen for Dual numbers as GenericLinearAlgebra lacks range support
        if eltype(data) <: ForwardDiff.Dual
            D, V = eigen(Hermitian(data))
            n_eff = min(ds.n, length(D))
            return D[1:n_eff], [Ket(b, V[:, k]) for k=1:n_eff]
        else
            D, V = eigen(Hermitian(data), 1:ds.n)
            states = [Ket(b, V[:, k]) for k=1:length(D)]
            return D, states
        end
    else
        warning && @warn(nonhermitian_warning)
        D, V = eigen(data)
        states = [Ket(b, V[:, k]) for k=1:length(D)]
        perm = sortperm(D, by=real)
        permute!(D, perm)
        permute!(states, perm)
        n_eff = min(ds.n, length(D))
        return D[1:n_eff], states[1:n_eff]
    end
end

function eigenstates(op::Operator, ds::KrylovDiag; warning::Bool=true, kwargs...)
    b = basis(op)
    ishermitian(op) || (warning && @warn(nonhermitian_warning))
    if ds.v0 === nothing
        D, Vs = eigsolve(op.data, ds.n, :SR; krylovdim = ds.krylovdim, kwargs...)
    else
        D, Vs = eigsolve(op.data, ds.v0, ds.n, :SR; krylovdim = ds.krylovdim, kwargs...)
    end
    states = [Ket(b, Vs[k]) for k=1:ds.n]
    D[1:ds.n], states
end

"""
    eigenenergies(op::AbstractOperator[, n::Int; warning=true, kwargs...])

Calculate the lowest n eigenvalues of given operator.
"""
function eigenenergies(op::AbstractOperator; kw...)
    ds, kw_rem = detect_diagstrategy(op; kw...)
    eigenenergies(op, ds; kw_rem...)
end

eigenenergies(op::AbstractOperator, n::Int; kw...) = eigenenergies(op; kw..., n=n)

function eigenenergies(op::Operator, ds::LapackDiag; warning=true)
    data = op.data
    if ishermitian(op)
        if eltype(data) <: ForwardDiff.Dual
            D = eigvals(Hermitian(data))
            n_eff = min(ds.n, length(D))
            return D[1:n_eff]
        else
            D = eigvals(Hermitian(data), 1:ds.n)
            return D
        end
    else
        warning && @warn(nonhermitian_warning)
        D = eigvals(data)
        sort!(D, by=real)
        n_eff = min(ds.n, length(D))
        return D[1:n_eff]
    end
end

eigenenergies(op::Operator, ds::DiagStrategy; kwargs...) = eigenstates(op, ds; kwargs...)[1]

"""
    simdiag(ops; atol, rtol)

Simultaneously diagonalize commuting Hermitian operators specified in `ops`.
"""
function simdiag(ops::Vector{<:AbstractOperator}; atol::Real=1e-14, rtol::Real=1e-14)
    for A=ops
        if !ishermitian(A)
            error("Non-hermitian operator given!")
        end
    end

    # Sum using generator to preserve Dual types during AD
    combined_data = sum(op.data for op in ops)
    if combined_data isa AbstractSparseMatrix
        combined_data = Array(combined_data)
    end
    
    # Use Hermitian for AD stability; requires GenericLinearAlgebra for Dual support
    d, v = eigen(Hermitian(combined_data))

    # Preserve Dual types in eigenvalues
    evals = [similar(d, length(d)) for i=1:length(ops)]
    
    for i=1:length(ops)
        op_data = ops[i].data
        evals[i] .= real.(diag(v' * op_data * v))
        
        # AD Fix: Verification check on values only to ignore derivative noise
        v_val = ForwardDiff.value.(v)
        op_val = ForwardDiff.value.(op_data)
        ev_val = ForwardDiff.value.(evals[i])
        
        if !isapprox(op_val * v_val, v_val * diagm(ev_val); atol=atol, rtol=rtol)
            error("Simultaneous diagonalization failed!")
        end
    end

    index = sortperm(real(evals[1][:]))
    evals_sorted = [real(evals[i][index]) for i=1:length(ops)]
    return evals_sorted, v[:, index]
end
