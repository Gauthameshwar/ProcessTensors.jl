# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/choi.jl
# Contributor: John F. Kam
#
# Converts Liouville / vectorized process tensor MPOs to Choi MPOs via local
# tensor re-indexing.

import ITensorMPS: MPO as CoreMPO

"""
    choi_pt_mpo(pt) -> MPO{Hilbert}

Convert a process tensor MPO to its unnormalized Choi MPO.

Each core has separate dimension-``d`` `in` and `out` Hilbert indices,
tagged `"Site,Choi,in,tstep=k"` and `"Site,Choi,out,tstep=k"`, and their
primed partners. Standard MPO ordering is retained: primed rows (bra-facing),
unprimed columns (ket-facing). The `in`/`out` tags label temporal ports;
priming does not label time. Memory links and `pt` are unchanged.
Use `siteinds(W, j; tags="in", plev=0)` to select an input index.

# Examples
```julia
W = choi_pt_mpo(pt)
inputs = siteinds(W, 1; tags="in", plev=0)
```
"""
function choi_pt_mpo(pt::ProcessTensor)
    cores = ITensor[]

    for k in 0:(pt.nsteps - 1)
        in_k = only(input_sites(pt, k))
        out_k = only(output_sites(pt, k))

        d = isqrt(dim(in_k))
        d * d == dim(in_k) == dim(out_k) || throw(
            ArgumentError(
                "choi_pt_mpo: input and output legs at tstep=$k must have the same perfect-square dimension.",
            ),
        )

        # Define constituent Hilbert indices.
        h_in = Index(d; tags="Site,Choi,in,tstep=$k")
        h_out = Index(d; tags="Site,Choi,out,tstep=$k")

        # Split Liouville legs into the primed and unprimed Hilbert indices via combiners.
        C_in = combiner(prime(h_in), h_in)
        C_out = combiner(prime(h_out), h_out)

        C_in = replaceind(C_in, combinedind(C_in), in_k)
        C_out = replaceind(C_out, combinedind(C_out), out_k)

        # Construct the re-indexed MPO core via contraction with combiners.
        core = pt.core[k + 1] * dag(C_in) * dag(C_out)

        push!(cores, core)
    end

    return MPO{Hilbert}(CoreMPO(cores))
end
