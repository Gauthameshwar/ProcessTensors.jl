using ProcessTensors
using ITensors
using LinearAlgebra
using Test

function _liouv_tensor_array(L_mpo, c)
    return Array(L_mpo[1], prime(c), c)
end

@testset "liouvillian.jl: method coverage for MPO_Liouville constructors" begin
    physical_sites = siteinds("S=1/2", 1)
    sL = liouv_sites(physical_sites)

    os_H = OpSum()
    os_H += 0.7, "Sz", 1

    tuple_jump_single = (0.25, "S-", 1)
    tuple_jump_vec = [tuple_jump_single]

    L1 = OpSum()
    L1 += 0.25, "S-", 1
    L2 = OpSum()
    L2 += 0.05, "S+", 1
    L3 = OpSum()
    L3 += 0.1, "Sz", 1

    @testset "keyword jump_ops on liouville sites" begin
        L_mpo = @test_nowarn MPO_Liouville(os_H, sL; jump_ops=tuple_jump_vec)
        @test length(L_mpo) == 1
    end

    @testset "keyword jump_ops default empty on liouville and physical sites" begin
        L_liouv = @test_nowarn MPO_Liouville(os_H, sL; jump_ops=[])
        L_phys = @test_nowarn MPO_Liouville(os_H, physical_sites; jump_ops=[])
        @test length(L_liouv) == 1
        @test length(L_phys) == 1
    end

    @testset "positional tuple jump operators" begin
        L_from_vec_liouv = @test_nowarn MPO_Liouville(os_H, tuple_jump_vec, sL)
        L_from_single_liouv = @test_nowarn MPO_Liouville(os_H, tuple_jump_single, sL)
        L_from_vec_phys = @test_nowarn MPO_Liouville(os_H, tuple_jump_vec, physical_sites)
        L_from_single_phys = @test_nowarn MPO_Liouville(os_H, tuple_jump_single, physical_sites)

        @test length(L_from_vec_liouv) == 1
        @test length(L_from_single_liouv) == 1
        @test length(L_from_vec_phys) == 1
        @test length(L_from_single_phys) == 1

        err = norm(_liouv_tensor_array(L_from_vec_liouv, sL[1]) - _liouv_tensor_array(L_from_single_liouv, sL[1]))
        @test err < 1e-12
    end

    @testset "OpSum jump operators as vector or varargs" begin
        L_from_vector_liouv = @test_nowarn MPO_Liouville(os_H, [L1, L2, L3], sL)
        L_from_varargs_liouv = @test_nowarn MPO_Liouville(os_H, L1, L2, L3, sL)
        L_from_varargs_phys = @test_nowarn MPO_Liouville(os_H, L1, L2, L3, physical_sites)

        @test length(L_from_vector_liouv) == 1
        @test length(L_from_varargs_liouv) == 1
        @test length(L_from_varargs_phys) == 1

        err = norm(_liouv_tensor_array(L_from_vector_liouv, sL[1]) - _liouv_tensor_array(L_from_varargs_liouv, sL[1]))
        @test err < 1e-12
    end

    @testset "unsupported 4-argument physical+liouville tuple call" begin
        @test_throws MethodError MPO_Liouville(os_H, tuple_jump_vec, physical_sites, sL)
    end

    @testset "output site indices: liouv_sites vs physical_sites input" begin
        # When passing liouv_sites, the output MPO should use the same site indices (by ID)
        L_with_liouv = MPO_Liouville(os_H, sL; jump_ops=tuple_jump_vec)
        inds_liouv = siteinds(L_with_liouv)
        @test length(inds_liouv) == 1  # Single site system
        # siteinds returns tuples of (bra_idx, ket_idx); extract the ket_idx (second element)
        all_liouv_flat = collect(Iterators.flatten(inds_liouv))
        ket_idx_liouv = filter(x -> plev(x)==0, all_liouv_flat)
        @test length(ket_idx_liouv) == 1
        @test ket_idx_liouv[1] == sL[1]

        # When passing physical_sites, the output MPO creates NEW internal liouv_sites
        L_with_phys = MPO_Liouville(os_H, physical_sites; jump_ops=tuple_jump_vec)
        inds_phys = siteinds(L_with_phys)
        @test length(inds_phys) == 1  # Single site system
        # siteinds returns tuples of (bra_idx, ket_idx); extract the ket_idx (second element)
        all_phys_flat = collect(Iterators.flatten(inds_phys))
        ket_idx_phys = filter(x -> plev(x)==0, all_phys_flat)
        @test length(ket_idx_phys) == 1
        @test ket_idx_phys[1] != sL[1]
        @test ket_idx_phys[1] != physical_sites[1]
        @test dim(ket_idx_phys[1]) == dim(physical_sites[1])^2

        # Verify dimensions match expected Liouville space
        @test dim(ket_idx_liouv[1]) == 4
        @test dim(ket_idx_phys[1]) == 4
    end

    @testset "output site indices: 2-site system" begin
        # Set up 2-site system
        physical_sites_2 = siteinds("S=1/2", 2)
        sL_2 = liouv_sites(physical_sites_2)

        os_H_2 = OpSum()
        os_H_2 += 0.5, "Sz", 1
        os_H_2 += 0.3, "Sx", 2
        os_H_2 += 0.7, "Sz", 1, "Sz", 2

        tuple_jump_2 = [(0.1, "S-", 1), (0.2, "S+", 2)]

        # Test with liouv_sites
        L_2_liouv = MPO_Liouville(os_H_2, sL_2; jump_ops=tuple_jump_2)
        inds_2_liouv = siteinds(L_2_liouv)
        @test length(inds_2_liouv) == 2  # Two site system
        all_2_liouv_flat = collect(Iterators.flatten(inds_2_liouv))
        ket_idx_2_liouv = filter(x -> plev(x)==0, all_2_liouv_flat)
        @test length(ket_idx_2_liouv) == 2
        @test all(ket_idx_2_liouv .== sL_2)

        # Test with physical_sites (creates new internal Liouville indices)
        L_2_phys = MPO_Liouville(os_H_2, physical_sites_2; jump_ops=tuple_jump_2)
        inds_2_phys = siteinds(L_2_phys)
        @test length(inds_2_phys) == 2  # Two site system
        all_2_phys_flat = collect(Iterators.flatten(inds_2_phys))
        ket_idx_2_phys = filter(x -> plev(x)==0, all_2_phys_flat)
        @test length(ket_idx_2_phys) == 2
        
        # New indices should differ from input pre-created ones
        @test all(ket_idx_2_phys .!= sL_2)
        
        # New indices should differ from physical sites
        @test all(ket_idx_2_phys .!= physical_sites_2)
        
        # All dimensions should be d² = 4
        for i in 1:2
            @test all(dim.(ket_idx_2_liouv) .== 4)
        end
    end
end
