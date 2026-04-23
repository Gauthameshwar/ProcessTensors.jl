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

# ------------------------ Density Matrix Constructors ---------------------

function to_dm(ψ::AbstractMPS{Hilbert})::MPO{Hilbert}
    return outer(ψ', ψ)
end

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

# --------------------- Vectorization and Unvectorization ---------------------

function to_liouville(ψ::AbstractMPS{Hilbert})::MPS{Liouville}
    return to_liouville(to_dm(ψ))
end


function to_liouville(
    ρ::AbstractMPO{Hilbert}, 
    sites::Union{Nothing, AbstractVector{<:Index}},
)::MPS{Liouville}
    """
    Vectorize a density matrix from Hilbert space to Liouville space.
    
    **Arguments:**
    - `ρ::AbstractMPO{Hilbert}`: Hilbert-space density matrix
    - `sites::Vector{Index}`: Pre-created Liouville sites (dimension d² for each site).
      If provided, these indices are used for vectorization. If `nothing`, new indices are created.
    
    **Returns:** `MPS{Liouville}` with vectorized density matrix. The Liouville sites used are accessible
    via `siteinds(ρ_vec)` if no pre-created sites were used during creation.
    
    **Recommended workflow (for index consistency):**
    ```julia
    s = siteinds("S=1/2", N)           # Define physical sites once
    s_L = liouv_sites(s)               # Create Liouville sites once
    
    ρ_vec = to_liouville(density_mpo, s_L)  # Reuse same Liouville sites
    ```
    """
    return to_liouville(ρ; sites=sites)
end

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

function to_hilbert(ρ::AbstractMPS{Liouville})::MPO{Hilbert}
    """
    Unvectorize a density matrix from Liouville space back to Hilbert space (inverse of `to_liouville`).
    
    **Arguments:**
    - `ρ::AbstractMPS{Liouville}`: Liouville-space density matrix (vectorized)
    
    **Returns:** `MPO{Hilbert}` density matrix in standard Hilbert-space representation.
    
    The Hilbert-space MPO uses the physical indices recovered from the Liouville basis tags
    (canonically `ptype=...`, while still accepting legacy `phys=...` tags).
    """
    raw_tensors = [ρ.core[i] for i in eachindex(ρ.core)]
    unzipped_tensors = [raw_tensors[i] * ρ.combiners[i] for i in eachindex(raw_tensors)]
    return MPO{Hilbert}(CoreMPO(unzipped_tensors))
end

# --------------------- Liouvillian site Constructors ---------------------

"""
    liouv_sites(physical_sites::AbstractVector{<:Index}) -> Vector{Index}

Create Liouville-space sites from physical (Hilbert-space) sites.

Each Liouville site has dimension d² for a physical site of dimension d.

**Recommended workflow (BEST PRACTICE):**
```julia
# Define all sites at the beginning of your code
s = siteinds("S=1/2", N)          # Physical sites
s_L = liouv_sites(s)             # Liouville sites - CREATE ONCE!

# Pass s_L to ALL subsequent operations
ρ = to_dm(ψ)
ρ_vec = to_liouville(ρ; sites=s_L)       # Pass s_L

os_H = OpSum() + ...
L_mpo = MPO_Liouville(os_H, s, s_L)      # Pass both physical and Liouville sites

# All operations use the SAME indices - NO manual index matching needed!
result = apply(L_mpo, ρ_vec)
```

**Why this matters:**
- Each call to `liouv_sites(...)` creates completely new Index objects
- ITensors automatic contraction requires exact Index object identity
- If you create Liouville indices separately in `to_liouville` and `MPO_Liouville`, they won't match
- Reusing the same pre-created `s_L` keeps everything consistent (like ITensorMPS conventions)

**Arguments:**
- `physical_sites`: Vector of physical indices (typically from `siteinds(\"S=1/2\", N)`)

**Returns:** Vector of Liouville indices (one per site), tagged with `"Liouv"` and
the physical site family metadata (`"ptype=..."`).
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

# --------------------- Operator Constructions in the Liouville space ---------------------

# The types of embedding of the physical operators into the Liouville space, as determined by the operator name suffix.
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

# --------------------- Model Construction Helpers ---------------------
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

# --------------------- Liouvillian MPO Construction ---------------------
# Reject empty site inputs early, e.g. `MPO_Liouville(H, Index[])` is not meaningful.
function _assert_sites_nonempty(sites::AbstractVector{<:Index})
    isempty(sites) && throw(ArgumentError("MPO_Liouville: received an empty sites vector."))
    return nothing
end

# Reuse Liouville sites when given, or build them from physical sites, e.g. `sites=s` returns `liouv_sites(s)`.
function _liouv_sites_from_last_arg(sites::AbstractVector{<:Index})
    _assert_sites_nonempty(sites)
    all_liouv = all(_is_liouv_site, sites)
    any_liouv = any(_is_liouv_site, sites)
    any_liouv && !all_liouv && throw(
        ArgumentError("MPO_Liouville: mixed physical and Liouville indices detected in sites argument."),
    )
    return all_liouv ? sites : liouv_sites(sites)
end

# Normalize tuple jump operators, e.g. `(γ, "S-", 1)` becomes `[(γ, "S-", 1)]`.
function _normalize_tuple_jump_ops(jump_ops)
    if jump_ops isa _TUPLE_JUMP_TYPE
        γ, opname, site = jump_ops
        return [(γ, String(opname), Int(site))]
    elseif jump_ops isa AbstractVector
        isempty(jump_ops) && return Tuple{Number,String,Int}[]
        all(op -> op isa _TUPLE_JUMP_TYPE, jump_ops) || throw(
            ArgumentError(
                "MPO_Liouville: tuple jump operators must be (rate, op_name::String, site::Int).",
            ),
        )
        return [(γ, String(opname), Int(site)) for (γ, opname, site) in jump_ops]
    end
    throw(
        ArgumentError(
            "MPO_Liouville: unsupported tuple jump_ops format $(typeof(jump_ops)).",
        ),
    )
end

# Normalize Lindblad OpSum inputs, e.g. a single `OpSum` becomes a one-element vector.
function _normalize_lindblad_ops(jump_ops)
    if jump_ops isa OpSum
        return [jump_ops]
    elseif jump_ops isa AbstractVector{<:OpSum}
        isempty(jump_ops) && return OpSum[]
        all(op -> op isa OpSum, jump_ops) || throw(
            ArgumentError("MPO_Liouville: Lindblad jump operators must be OpSum objects."),
        )
        return collect(jump_ops)
    end
    throw(
        ArgumentError(
            "MPO_Liouville: unsupported Lindblad jump_ops format $(typeof(jump_ops)).",
        ),
    )
end

# Build the MPO from tuple jump operators, e.g. local `(γ, "S-", 1)` terms plus the Hamiltonian commutator.
function _build_liouvillian_mpo(os_H::OpSum, tuple_jump_ops::AbstractVector, sites::AbstractVector{<:Index}; splitblocks::Bool=true)
    liouv_sites_arg = _liouv_sites_from_last_arg(sites)
    os_Liouville = build_liouvillian_opsum(os_H, tuple_jump_ops)
    return MPO(os_Liouville, liouv_sites_arg; splitblocks=splitblocks)
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

# Build the MPO from Lindblad OpSum channels, e.g. `[OpSum() + (γ, "S-", 1)]` plus the Hamiltonian commutator.
function _build_liouvillian_mpo_from_lindblad(
    os_H::OpSum,
    os_lindblad::AbstractVector{<:OpSum},
    sites::AbstractVector{<:Index};
    splitblocks::Bool=true,
)
    liouv_sites_arg = _liouv_sites_from_last_arg(sites)
    os_Liouville = build_liouvillian_opsum_from_lindblad(os_H, os_lindblad)
    return MPO(os_Liouville, liouv_sites_arg; splitblocks=splitblocks)
end

function build_liouvillian_opsum_from_lindblad(
    os_H::OpSum,
    os_lindblad::AbstractVector{<:OpSum},
)
    """
    Build a Liouvillian OpSum from a Hamiltonian and a vector of Lindbladian jump operator OpSums.
    
    Each OpSum in os_lindblad should represent a single dissipation channel, i.e., a jump operator 
    L with its associated decay rate γ. The OpSum term should be of the form:
        os_jump_k += γ, "OpName", site1[, site2, ...]
    
    This function extracts γ and the operator L from each OpSum and builds the corresponding
    dissipator superoperator terms: γ(L* ⊗ L) - (γ/2)(I ⊗ L†L) - (γ/2)((L†L)^T ⊗ I).
    """
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
    OpSum_Liouville(os_H::OpSum; jump_ops=[])
    OpSum_Liouville(os_H::OpSum, jump_ops)

Generate the symbolic Liouvillian OpSum from a physical Hamiltonian and Lindblad jump operators,
**without** converting it into an MPO.

This is the recommended entry point for generating Trotter gates for TEBD in Liouville space,
or for any workflow where you need the symbolic operator sum before choosing a representation.

Accepts the same jump operator formats as `MPO_Liouville`:
- Tuple-based: `(γ, "S-", 1)` or `[(γ, "S-", 1), ...]`
- OpSum-based: `OpSum() + (γ, "S-", 1)` or `[OpSum() + (γ, "S-", 1), ...]`

# Examples

```julia
# For MPO construction:
L_mpo = MPO(OpSum_Liouville(os_H; jump_ops=jumps), s_L)

# For TEBD gate generation:
gates = ops(exp(dt * OpSum_Liouville(os_H; jump_ops=jumps); alg=Trotter{2}()), s_L)
```
"""
function OpSum_Liouville(os_H::OpSum; jump_ops=Tuple{Number,String,Int}[])
    return OpSum_Liouville(os_H, jump_ops)
end

function OpSum_Liouville(os_H::OpSum, jump_ops)
    if jump_ops isa OpSum || (jump_ops isa AbstractVector && !isempty(jump_ops) && all(op -> op isa OpSum, jump_ops))
        lindblad_ops = _normalize_lindblad_ops(jump_ops)
        return build_liouvillian_opsum_from_lindblad(os_H, lindblad_ops)
    end
    tuple_jump_ops = _normalize_tuple_jump_ops(jump_ops)
    return build_liouvillian_opsum(os_H, tuple_jump_ops)
end

"""
    MPO_Liouville(os_H::OpSum, sites::AbstractVector{<:Index}; jump_ops=[])

Build a Liouvillian MPO using either physical or pre-created Liouville sites.
"""
function MPO_Liouville(
    os_H::OpSum,
    sites::AbstractVector{<:Index};
    jump_ops=Tuple{Number,String,Int}[],
    splitblocks::Bool=true,
)
    if jump_ops isa OpSum || (jump_ops isa AbstractVector && !isempty(jump_ops) && all(op -> op isa OpSum, jump_ops))
        lindblad_ops = _normalize_lindblad_ops(jump_ops)
        return _build_liouvillian_mpo_from_lindblad(os_H, lindblad_ops, sites; splitblocks=splitblocks)
    end

    tuple_jump_ops = _normalize_tuple_jump_ops(jump_ops)
    return _build_liouvillian_mpo(os_H, tuple_jump_ops, sites; splitblocks=splitblocks)
end

"""
    MPO_Liouville(os_H::OpSum, jump_ops, sites::AbstractVector{<:Index}; splitblocks=true)

Build a Liouvillian MPO from positional jump operators and site indices.

Accepts:
- Tuple-based jump operators: single tuple or vector of tuples, e.g. (γ, "S-", 1) or [(γ, "S-", 1)]
- Lindblad OpSum operators: single OpSum or vector of OpSums, e.g. [OpSum() += γ, "S-", 1]
- For multiple varargs OpSums, use: MPO_Liouville(H, L1, L2, L3, sites)
"""
function MPO_Liouville(
    os_H::OpSum,
    jump_ops,
    sites::AbstractVector{<:Index};
    splitblocks::Bool=true,
)
    # Check if it's OpSum-based
    if jump_ops isa OpSum || (jump_ops isa AbstractVector && !isempty(jump_ops) && all(op -> op isa OpSum, jump_ops))
        lindblad_ops = _normalize_lindblad_ops(jump_ops)
        return _build_liouvillian_mpo_from_lindblad(os_H, lindblad_ops, sites; splitblocks=splitblocks)
    end

    # Otherwise treat as tuple-based
    tuple_jump_ops = _normalize_tuple_jump_ops(jump_ops)
    return _build_liouvillian_mpo(os_H, tuple_jump_ops, sites; splitblocks=splitblocks)
end

"""
    MPO_Liouville(os_H::OpSum, L1::OpSum, L2::OpSum, Lrest..., sites::AbstractVector{<:Index}; splitblocks=true)

Build a Liouvillian MPO from multiple Lindblad jump operators as varargs.

Each OpSum argument (L1, L2, L3, ...) is treated as one Lindblad dissipation channel.
The rightmost positional argument must be physical or Liouville sites.

**Note:** For a single OpSum jump operator, use the vector form:
`MPO_Liouville(os_H, [L1], sites)` or `MPO_Liouville(os_H, sites; jump_ops=[L1])`
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
    return _build_liouvillian_mpo_from_lindblad(os_H, lindblad_ops, sites; splitblocks=splitblocks)
end
