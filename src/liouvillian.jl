# src/liouvillian.jl
using ITensors, LinearAlgebra

const _LIOUV_PTYPE_PREFIX = "ptype="
const _LIOUV_LEGACY_PTYPE_PREFIX = "phys="

# Detect metadata tags that store the physical site family, e.g. `ptype=Boson` or legacy `phys=Boson`.
_is_liouv_ptype_tag(token::AbstractString) =
    startswith(token, _LIOUV_PTYPE_PREFIX) || startswith(token, _LIOUV_LEGACY_PTYPE_PREFIX)

# Recover the physical site family from Liouville tags, e.g. `["Liouv", "ptype=Fermion", "n=2"] -> "Fermion"`.
function _liouv_site_type(tag_tokens::AbstractVector{<:AbstractString})
    for prefix in (_LIOUV_PTYPE_PREFIX, _LIOUV_LEGACY_PTYPE_PREFIX)
        for token in tag_tokens
            if startswith(token, prefix)
                return String(token[length(prefix) + 1:end])
            end
        end
    end

    for token in tag_tokens
        if token == "Site" || token == "Liouv" || startswith(token, "n=") || startswith(token, "l=") || _is_liouv_ptype_tag(token)
            continue
        end
        return String(token)
    end
    return nothing
end

# Build canonical Liouville tags, e.g. `S=1/2,Site,n=3` becomes `Liouv,S=1/2,ptype=S=1/2,n=3`.
function _liouv_tagset(s::Index)
    tokens = tag_tokens(s)
    site_type = _liouv_site_type(tokens)
    site_type === nothing && throw(
        ArgumentError("liouv_sites: Could not infer a physical SiteType tag from index $(s)."),
    )

    rest = filter(t -> t != "Liouv" && t != site_type && !_is_liouv_ptype_tag(t), tokens)
    return ITensors.TagSet(join(vcat(["Liouv", "$(_LIOUV_PTYPE_PREFIX)$site_type"], rest), ","))
end

# Check whether an index is already Liouville-tagged, e.g. `Liouv,ptype=Boson,n=1`.
_is_liouv_site(s::Index) = has_tag_token(s, "Liouv")

_TUPLE_JUMP_TYPE = Tuple{<:Number,<:AbstractString,<:Integer}

"""
    to_dm(ψ::AbstractMPS{Hilbert})
    to_dm(ψs::AbstractVector{<:AbstractMPS{Hilbert}}; coeffs)

Construct a Hilbert-space density MPO.

For one state, returns the pure density operator ``|ψ⟩⟨ψ|``. For a vector of
states, returns the classical mixture ``∑ᵢ pᵢ |ψᵢ⟩⟨ψᵢ|`` with non-negative
`coeffs` summing to one.

# Examples
```julia
ψ1 = random_mps(sites)
ψ2 = random_mps(sites)
ρ_pure = to_dm(ψ1)
ρ_mixed = to_dm([ψ1, ψ2]; coeffs=[0.3, 0.7])
```
"""
function to_dm(ψ::AbstractMPS{Hilbert})::MPO{Hilbert}
    return outer(ψ', ψ)
end

"""
    to_dm(ψs::AbstractVector{<:AbstractMPS{Hilbert}}; coeffs)

Construct the mixed Hilbert-space density MPO
`sum(coeffs[i] * |ψ_i><ψ_i| for i in eachindex(ψs))`.

`coeffs` must be non-negative, have the same length as `ψs`, and sum to one.
"""
function to_dm(
    ψarr::AbstractVector{<:AbstractMPS{Hilbert}};
    coeffs::AbstractVector{<:Real}=fill(1 / length(ψarr), length(ψarr)),
)::MPO{Hilbert}
    length(ψarr) == length(coeffs) || throw(
        ArgumentError(
            "to_dm: Number of state vectors and coefficients must match. Got $(length(ψarr)) and $(length(coeffs)).",
        ),
    )
    sum(coeffs) ≈ 1 || throw(
        ArgumentError("to_dm: Coefficients must sum to 1. Got sum(coeffs) = $(sum(coeffs))."),
    )
    all(c -> c >= 0, coeffs) || throw(DomainError(coeffs, "to_dm: Coefficients must be non-negative."))

    ρ = to_dm(ψarr[1]) * float(coeffs[1])
    for i in 2:length(ψarr)
        ρ += to_dm(ψarr[i]) * float(coeffs[i])
    end
    return ρ
end

"""
    to_liouville(ψ::AbstractMPS{Hilbert})

Convert a Hilbert-space pure state by forming `to_dm(ψ)` and vectorizing the
resulting density MPO.
"""
function to_liouville(ψ::AbstractMPS{Hilbert})::MPS{Liouville}
    return to_liouville(to_dm(ψ))
end

function to_liouville(
    ρ::AbstractMPO{Hilbert},
    sites::Union{Nothing, AbstractVector{<:Index}},
)::MPS{Liouville}
    return to_liouville(ρ; sites=sites)
end

"""
    to_liouville(ρ::AbstractMPO{Hilbert}; sites=nothing)
    to_liouville(ψ::AbstractMPS{Hilbert})

Vectorize a Hilbert-space density operator into a Liouville-space MPS.

Each local Hilbert bra/ket pair is fused into one Liouville site of dimension
``d^2``. When `sites` is provided, those exact Liouville indices are reused so
density states, Liouvillian MPOs, instruments, and process-tensor legs contract
by index identity.

# Examples
```julia
s = siteinds("S=1/2", 4)
s_L = liouv_sites(s)
ρL = to_liouville(ρ; sites=s_L)
```
"""
function to_liouville(
    ρ::AbstractMPO{Hilbert};
    sites::Union{Nothing, AbstractVector{<:Index}}=nothing,
)::MPS{Liouville}
    phys_inds = siteinds(ρ)
    
    if sites === nothing
        # Create new Liouville sites internally (backward compatible)
        liouv_tags = [_liouv_tagset(phys_inds[i][1]) for i in eachindex(phys_inds)]
        combiners_ρ = [combiner(phys_inds[i]...; tags=liouv_tags[i]) for i in eachindex(phys_inds)]
        liouv_tensors = [ρ.core[i] * combiners_ρ[i] for i in eachindex(phys_inds)]
    else
        # Use provided Liouville sites
        length(sites) == length(phys_inds) || throw(
            ArgumentError(
                "to_liouville: Number of Liouville sites ($(length(sites))) must match "
                * "number of physical sites ($(length(phys_inds))).",
            ),
        )
        combiners_ρ = ITensor[]
        liouv_tensors = ITensor[]
        # This loop aligns each provided Liouville index with the local density tensor so dimensions and index identity match for later contractions.
        for i in eachindex(phys_inds)
            d2 = dim(sites[i])
            d = Int(round(sqrt(d2)))
            d * d == d2 || throw(
                ArgumentError(
                    "to_liouville: Provided Liouville site $i has dim=$d2, expected perfect square d².",
                ),
            )

            C = combiner(phys_inds[i]...; tags=tags(sites[i]))
            t = ρ.core[i] * C
            c_new = uniqueind(t, ρ.core[i])
            push!(combiners_ρ, replaceind(C, c_new, sites[i]))
            push!(liouv_tensors, replaceind(t, c_new, sites[i]))
        end
    end

    return MPS{Liouville}(CoreMPS(liouv_tensors), combiners_ρ)
end

"""
    to_hilbert(ρ::AbstractMPS{Liouville})

Unvectorize a Liouville-space density MPS into a Hilbert-space density MPO.

Uses the `combiners` stored on `ρ` to split each Liouville site back into the
corresponding Hilbert bra/ket site pair.

# Examples
```julia
ρ = to_hilbert(ρL)
```
"""
function to_hilbert(ρ::AbstractMPS{Liouville})::MPO{Hilbert}
    raw_tensors = [ρ.core[i] for i in eachindex(ρ.core)]
    unzipped_tensors = [raw_tensors[i] * ρ.combiners[i] for i in eachindex(raw_tensors)]
    return MPO{Hilbert}(CoreMPO(unzipped_tensors))
end

"""
    liouv_sites(physical_sites::AbstractVector{<:Index}) -> Vector{Index}

Construct Liouville-space site indices from Hilbert-space site indices.

Each returned index has dimension ``d^2`` for a physical site of dimension
``d`` and carries the `"Liouv"` tag plus physical site-family metadata
(`"ptype=..."`). Reuse the same objects across [`to_liouville`](@ref),
[`MPO_Liouville`](@ref), and process-tensor instruments.

# Examples
```julia
s_L = liouv_sites(siteinds("S=1/2", 4))
```
"""
function liouv_sites(physical_sites::AbstractVector{<:Index})
    liouv = [Index(dim(s)^2; tags=_liouv_tagset(s)) for s in physical_sites]
    return liouv
end

# Rebuild the temporary physical index used by `op`, e.g. `Liouv,ptype=Electron,n=4` gives `Electron,Site,n=4`.
function _phys_site_from_liouv(s::Index)
    tokens = tag_tokens(s)
    site_type = _liouv_site_type(tokens)
    site_type === nothing && throw(
        ArgumentError(
            "Could not infer physical SiteType for Liouville index $(s). "
            * "Create Liouville sites with liouv_sites(physical_sites), or pass Liouville indices tagged with `ptype=SiteType`.",
        ),
    )

    d2 = dim(s)
    d = isqrt(d2)
    d * d == d2 || throw(
        ArgumentError(
            "Liouville index $(s) has dim=$d2, expected a perfect square d² to recover the local physical dimension.",
        ),
    )

    n_token = findfirst(t -> startswith(t, "n="), tokens)
    phys_tags = n_token === nothing ? ITensors.TagSet(join([site_type, "Site"], ",")) :
                ITensors.TagSet(join([site_type, "Site", tokens[n_token]], ","))
    return Index(d; tags=phys_tags)
end

# Physical Hilbert sites from a density matrix MPO (unprimed leg at each site).
function _phys_sites_from_hilbert_mpo(mpo::AbstractMPO{Hilbert})
    return Index[
        only(filter(i -> plev(i) == 0, inds(mpo.core[j])))
        for j in eachindex(mpo.core)
    ]
end

_phys_sites_from_hilbert_state(ρ::AbstractMPO{Hilbert}) = _phys_sites_from_hilbert_mpo(ρ)
_phys_sites_from_hilbert_state(ψ::AbstractMPS{Hilbert}) = siteinds(ψ)

# The operator-name suffix determines how a physical operator is embedded in Liouville space.
abstract type _LiouvSideTrait end
struct _LiouvLeft <: _LiouvSideTrait end
struct _LiouvRight <: _LiouvSideTrait end
struct _LiouvJump <: _LiouvSideTrait end
struct _LiouvLdagLLeft <: _LiouvSideTrait end
struct _LiouvLdagLRight <: _LiouvSideTrait end

# Split a Liouville operator name into its physical part and action side, e.g. `Sz_R` -> (`Sz`, right action).
function _parse_liouv_op(op_name::String)
    if endswith(op_name, "_LdagL_L")
        return op_name[1:end-8], _LiouvLdagLLeft()
    elseif endswith(op_name, "_LdagL_R")
        return op_name[1:end-8], _LiouvLdagLRight()
    elseif endswith(op_name, "_Jump")
        return op_name[1:end-5], _LiouvJump()
    elseif endswith(op_name, "_L")
        return op_name[1:end-2], _LiouvLeft()
    elseif endswith(op_name, "_R")
        return op_name[1:end-2], _LiouvRight()
    end
    return nothing, nothing
end

# Embed a physical matrix into Liouville space, e.g. left action maps `A` to `I ⊗ A`.
_superop_matrix(::_LiouvLeft, A::AbstractMatrix, Id::AbstractMatrix) = kron(Id, A)
_superop_matrix(::_LiouvRight, A::AbstractMatrix, Id::AbstractMatrix) = kron(transpose(A), Id)
_superop_matrix(::_LiouvJump, A::AbstractMatrix, Id::AbstractMatrix) = kron(conj(A), A)

function _superop_matrix(::_LiouvLdagLLeft, A::AbstractMatrix, Id::AbstractMatrix)
    LdagL = A' * A
    return kron(Id, LdagL)
end

function _superop_matrix(::_LiouvLdagLRight, A::AbstractMatrix, Id::AbstractMatrix)
    LdagL = A' * A
    return kron(transpose(LdagL), Id)
end

# Materialize a physical operator as a dense matrix, e.g. `Sz` on an `S=1/2` site becomes a `2×2` array.
function _op_matrix(op_name::AbstractString, s_phys::Index; kwargs...)
    opT = ITensors.op(op_name, s_phys; kwargs...)
    return Array(opT, prime(s_phys), s_phys)
end

function ITensors.op(on::ITensors.OpName{N}, ::ITensors.SiteType"Liouv", s::Index; kwargs...) where {N}
    op_name = String(N)
    s_phys = _phys_site_from_liouv(s)
    d = dim(s_phys)
    Id = Matrix{ComplexF64}(I, d, d)

    if op_name == "Id"
        return ITensor(Matrix{ComplexF64}(I, dim(s), dim(s)), prime(s), s)
    end

    phys_op, side = _parse_liouv_op(op_name)
    side === nothing && throw(ArgumentError("Operator $op_name not recognized for SiteType\"Liouv\"."))

    A = _op_matrix(phys_op, s_phys; kwargs...)
    return ITensor(_superop_matrix(side, A, Id), prime(s), s)
end

# Canonical OpSum builder: normalized tuple-jump vector
function _liouvillian_opsum(
    os_H::OpSum,
    tuple_jumps::AbstractVector{<:Tuple{<:Number,<:AbstractString,<:Integer}},
)
    jumps = [(γ, String(opname), Int(site)) for (γ, opname, site) in tuple_jumps]
    return build_liouvillian_opsum(os_H, jumps)
end

# Canonical OpSum builder: normalized Lindblad OpSum channel vector.
function _liouvillian_opsum(os_H::OpSum, lindblad_ops::AbstractVector{<:OpSum})
    return build_liouvillian_opsum_from_lindblad(os_H, lindblad_ops)
end

# Canonical MPO builder from a Liouvillian OpSum and physical or Liouville sites.
function _liouvillian_mpo_from_opsum(
    os_L::OpSum,
    sites::AbstractVector{<:Index};
    splitblocks::Bool=true,
)
    isempty(sites) && throw(ArgumentError("MPO_Liouville: received an empty sites vector."))
    all_liouv = all(_is_liouv_site, sites)
    any_liouv = any(_is_liouv_site, sites)
    any_liouv && !all_liouv && throw(
        ArgumentError("MPO_Liouville: mixed physical and Liouville indices detected in sites argument."),
    )
    liouv_sites_arg = all_liouv ? sites : liouv_sites(sites)
    return MPO{Liouville}(ITensor[], os_L, liouv_sites_arg; splitblocks=splitblocks)
end

# Build the Liouvillian OpSum from a Hamiltonian OpSum and a vector of jump operators (with rates).
function build_liouvillian_opsum(os_H::OpSum, jump_ops::AbstractVector{<:Tuple{<:Number,<:AbstractString,<:Integer}})
    os_L = OpSum()

    for term in ITensors.terms(os_H)
        c = ITensors.coefficient(term)
        ops = collect(last(term.args))

        left_args = Any[]
        for o in ops
            push!(left_args, "$(ITensors.name(o))_L")
            append!(left_args, collect(ITensors.sites(o)))
        end
        os_L += (-1im * c, left_args...)

        right_args = Any[]
        for o in ops
            push!(right_args, "$(ITensors.name(o))_R")
            append!(right_args, collect(ITensors.sites(o)))
        end
        os_L += (1im * c, right_args...)
    end

    for (gamma, op_name, site) in jump_ops
        os_L += (gamma, "$(op_name)_Jump", Int(site))
        os_L += (-gamma / 2, "$(op_name)_LdagL_L", Int(site))
        os_L += (-gamma / 2, "$(op_name)_LdagL_R", Int(site))
    end

    return os_L
end

# Build a Liouvillian OpSum from Lindblad-channel OpSums, each representing one
# jump operator with its rate in the term coefficient.
function build_liouvillian_opsum_from_lindblad(
    os_H::OpSum,
    os_lindblad::AbstractVector{<:OpSum},
)
    os_L = OpSum()

    # Add Hamiltonian commutator terms: -i[H, ·]
    for term in ITensors.terms(os_H)
        c = ITensors.coefficient(term)
        ops = collect(last(term.args))

        left_args = Any[]
        for o in ops
            push!(left_args, "$(ITensors.name(o))_L")
            append!(left_args, collect(ITensors.sites(o)))
        end
        os_L += (-1im * c, left_args...)

        right_args = Any[]
        for o in ops
            push!(right_args, "$(ITensors.name(o))_R")
            append!(right_args, collect(ITensors.sites(o)))
        end
        os_L += (1im * c, right_args...)
    end

    # Add dissipator terms for each Lindbladian jump operator OpSum
    for os_jump in os_lindblad
        for term in ITensors.terms(os_jump)
            gamma = ITensors.coefficient(term)
            ops = collect(last(term.args))

            # Jump operator: γ(L* ⊗ L)
            jump_args = Any[]
            for o in ops
                push!(jump_args, "$(ITensors.name(o))_Jump")
                append!(jump_args, collect(ITensors.sites(o)))
            end
            os_L += (gamma, jump_args...)

            # Left anticommutator: -(γ/2)(I ⊗ L†L)
            ldagl_left_args = Any[]
            for o in ops
                push!(ldagl_left_args, "$(ITensors.name(o))_LdagL_L")
                append!(ldagl_left_args, collect(ITensors.sites(o)))
            end
            os_L += (-gamma / 2, ldagl_left_args...)

            # Right anticommutator: -(γ/2)((L†L)^T ⊗ I)
            ldagl_right_args = Any[]
            for o in ops
                push!(ldagl_right_args, "$(ITensors.name(o))_LdagL_R")
                append!(ldagl_right_args, collect(ITensors.sites(o)))
            end
            os_L += (-gamma / 2, ldagl_right_args...)
        end
    end

    return os_L
end

"""
    OpSum_Liouville(os_H::OpSum; jump_ops=Tuple{Number,String,Int}[])
    OpSum_Liouville(os_H::OpSum, jump_ops)
    OpSum_Liouville(os_H::OpSum, L::OpSum)
    OpSum_Liouville(os_H::OpSum, Ls::AbstractVector{<:OpSum})

Construct the Liouville-space `OpSum` superoperator ``L`` for the master equation

```math
\\frac{d\\rho}{dt} = -i[H,\\rho] + \\sum_k \\gamma_k \\, \\mathcal{D}[L_k]\\rho,
```

where ``\\mathcal{D}[L]\\rho = L\\rho L^\\dagger - \\tfrac{1}{2}\\{L^\\dagger L, \\rho\\}``
is the Lindblad dissipator.

The Hamiltonian part is encoded as the commutator superoperator
``-i[H,\\cdot] = -i(H\\otimes I - I\\otimes H^\\top)`` on vectorized density matrices.
Each tuple jump ``(\\gamma, \\text{opname}, j)`` adds
``\\gamma\\,\\mathcal{D}[L_j]`` with ``L_j = \\text{op}(\\text{opname}, j)``.
For example, `jump_ops=[(0.1, "S-", 1)]` adds amplitude damping via
``L = S_-`` at site `1` with rate ``\\gamma = 0.1``.

Jump operators may also be supplied as `OpSum`s or vectors of `OpSum`s.

# Examples
```julia
H = OpSum()
H += 1.0, "Sz", 1
L = OpSum_Liouville(H; jump_ops=[(0.1, "S-", 1)])
```
"""
function OpSum_Liouville(os_H::OpSum; jump_ops=Tuple{Number,String,Int}[])
    return OpSum_Liouville(os_H, jump_ops)
end

"""
    OpSum_Liouville(os_H::OpSum, jump::Tuple{<:Number,<:AbstractString,<:Integer})

Construct the Liouvillian `OpSum` using one tuple-form Lindblad jump operator
such as `(γ, "S-", 1)`.
"""
OpSum_Liouville(os_H::OpSum, jump::_TUPLE_JUMP_TYPE) = begin
    γ, opname, site = jump
    _liouvillian_opsum(os_H, [(γ, String(opname), Int(site))])
end

"""
    OpSum_Liouville(os_H::OpSum, jumps::AbstractVector{<:Tuple{<:Number,<:AbstractString,<:Integer}})

Construct the Liouvillian `OpSum` using several tuple-form Lindblad jump
operators.
"""
OpSum_Liouville(
    os_H::OpSum,
    jumps::AbstractVector{<:Tuple{<:Number,<:AbstractString,<:Integer}},
) = _liouvillian_opsum(os_H, jumps)

"""
    OpSum_Liouville(os_H::OpSum, L::OpSum)

Construct the Liouvillian `OpSum` using one Lindblad jump operator written as
an `OpSum`.
"""
OpSum_Liouville(os_H::OpSum, L::OpSum) = _liouvillian_opsum(os_H, OpSum[L])

"""
    OpSum_Liouville(os_H::OpSum, Ls::AbstractVector{<:OpSum})

Construct the Liouvillian `OpSum` using several Lindblad jump operators written
as `OpSum`s.
"""
OpSum_Liouville(os_H::OpSum, Ls::AbstractVector{<:OpSum}) = _liouvillian_opsum(os_H, Ls)

# Untyped empty vector (e.g. `jump_ops=[]`) is treated as no jump operators.
OpSum_Liouville(os_H::OpSum, jump_ops::AbstractVector) = isempty(jump_ops) ?
    _liouvillian_opsum(os_H, Tuple{Number,String,Int}[]) :
    throw(ArgumentError("OpSum_Liouville: unsupported jump_ops format $(typeof(jump_ops))."))

function OpSum_Liouville(os_H::OpSum, jump_ops)
    throw(ArgumentError("OpSum_Liouville: unsupported jump_ops format $(typeof(jump_ops))."))
end

"""
    MPO_Liouville(os_H::OpSum, sites::AbstractVector{<:Index}; jump_ops=Tuple{Number,String,Int}[], splitblocks=true)
    MPO_Liouville(os_H::OpSum, jump_ops, sites::AbstractVector{<:Index}; splitblocks=true)

Construct a Liouville-space MPO from a Hamiltonian and Lindblad jump operators.

`sites` may be physical Hilbert-space sites or pre-created Liouville sites from
[`liouv_sites`](@ref). Reuse the same Liouville indices across
[`to_liouville`](@ref) and process-tensor objects.

# Examples
```julia
s_L = liouv_sites(sites)
L_mpo = MPO_Liouville(H, s_L; jump_ops=[(0.1, "S-", 1)])
```
"""
MPO_Liouville(
    os_H::OpSum,
    sites::AbstractVector{<:Index};
    jump_ops=Tuple{Number,String,Int}[],
    splitblocks::Bool=true,
) = _liouvillian_mpo_from_opsum(OpSum_Liouville(os_H, jump_ops), sites; splitblocks=splitblocks)

"""
    MPO_Liouville(os_H::OpSum, jump_ops, sites::AbstractVector{<:Index}; splitblocks=true)

Construct a Liouville-space MPO using supported tuple-form or `OpSum` Lindblad
jump operators supplied positionally.
"""
MPO_Liouville(
    os_H::OpSum,
    jump_ops,
    sites::AbstractVector{<:Index};
    splitblocks::Bool=true,
) = _liouvillian_mpo_from_opsum(OpSum_Liouville(os_H, jump_ops), sites; splitblocks=splitblocks)

"""
    MPO_Liouville(os_H::OpSum, L1::OpSum, L2::OpSum, Lrest..., sites::AbstractVector{<:Index}; splitblocks=true)

Construct a Liouville-space MPO using two or more Lindblad jump operators
written as positional `OpSum` arguments.
"""
function MPO_Liouville(
    os_H::OpSum,
    L1::OpSum,
    L2::OpSum,
    args...;
    splitblocks::Bool=true,
)
    isempty(args) && throw(
        ArgumentError(
            "MPO_Liouville: expected a final sites argument, e.g. MPO_Liouville(H, L1, L2, sites).",
        ),
    )

    sites = args[end]
    sites isa AbstractVector{<:Index} || throw(
        ArgumentError(
            "MPO_Liouville: last positional argument must be physical sites or Liouville sites, got $(typeof(sites)).",
        ),
    )

    extra_ops = args[1:end-1]
    all(op -> op isa OpSum, extra_ops) || throw(
        ArgumentError("MPO_Liouville: all arguments between H and sites must be OpSum jump operators."),
    )

    lindblad_ops = OpSum[L1, L2]
    append!(lindblad_ops, extra_ops)
    return _liouvillian_mpo_from_opsum(_liouvillian_opsum(os_H, lindblad_ops), sites; splitblocks=splitblocks)
end
