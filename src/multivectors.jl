
#   This file is part of Clifford.jl. It is licensed under the AGPL license
#   Clifford Copyright (C) 2019 Michael Reed

## SparseChain{V,G}

"""
    SparseChain{V,G} <: TensorGraded{V,G}

Sparse chain type with pseudoscalar `V::Manifold` and grade/rank `G::Int`.
"""
struct SparseChain{V,G,T} <: TensorGraded{V,G}
    v::SparseVector{T,Int}
end

SparseChain{V,G}(v::SparseVector{T,Int}) where {V,G,T} = SparseChain{V,G,T}(v)

for Vec ∈ (:(Values{L,T}),:(SubArray{T,1,Values{L,T}}))
    @eval function chainvalues(V::Manifold,m::$Vec,::Val{G}) where {G,L,T}
        N = rank(V); bng = binomial(N,G)
        G∉(0,N) && sum(m .== 0)/bng < fill_limit && (return Chain{V,G,T}(m))
        out = spzeros(T,bng)
        for i ∈ 1:bng
            @inbounds m[i] ≠ 0 && (out[i] = m[i])
        end
        SparseChain{V,G}(out) #Simplex{V,G,getbasis(V,@inbounds indexbasis(N,G)[out.nzind[1]]),T}(@inbounds m[out.nzind[1]])
    end
end

SparseChain{V,G}(m::Chain{V,G,T}) where {V,G,T} = chainvalues(V,value(m),Val{G}())
SparseChain{V}(m::Chain{V,G,T}) where {V,G,T} = SparseChain{V,G}(m)
SparseChain(m::Chain{V,G,T}) where {V,G,T} = SparseChain{V,G}(m)
SparseChain{V}(v::Vector{<:TensorTerm{V,G}}) where {V,G} = SparseChain{V,G}(sparsevec(bladeindex.(mdims(V),bits.(v)),value.(v),mdims(V)))
SparseChain(v::T) where T <: TensorTerm = v

function show(io::IO, m::SparseChain{V,G,T}) where {V,G,T}
    ib = indexbasis(mdims(V),G)
    o = m.v.nzind[1]
    @inbounds if T == Any && typeof(m.v[o]) ∈ parsym
        @inbounds tmv = typeof(m.v[o])
        par = (!(tmv<:TensorTerm)) && |(broadcast(<:,tmv,parval)...)
        @inbounds par ? print(io,m.v[o]) : print(io,"(",m.v[o],")")
    else
        @inbounds print(io,m.v[o])
    end
    @inbounds Leibniz.printindices(io,V,ib[o])
    length(m.v.nzind)>1 && for k ∈ m.v.nzind[2:end]
        @inbounds mvs = m.v[k]
        tmv = typeof(mvs)
        if |(broadcast(<:,tmv,parsym)...)
            par = (!(tmv<:TensorTerm)) && |(broadcast(<:,tmv,parval)...)
            par ? print(io," + (",mvs,")") : print(io," + ",mvs)
        else
            sbm = signbit(mvs)
            print(io,sbm ? " - " : " + ",sbm ? abs(mvs) : mvs)
        end
        @inbounds Leibniz.printindices(io,V,ib[k])
    end
end

==(a::SparseChain{V,G},b::SparseChain{V,G}) where {V,G} = prod(terms(a) .== terms(b))
==(a::SparseChain{V},b::SparseChain{V}) where V = iszero(a) && iszero(b)
==(a::SparseChain{V},b::T) where T<:TensorTerm{V} where V = false
==(a::T,b::SparseChain{V}) where T<:TensorTerm{V} where V = false

## MultiGrade{V,G}

@computed struct MultiGrade{V,G} <: TensorMixed{V}
    v::Values{count_ones(G),TensorGraded{V}}
end

@doc """
    MultiGrade{V,G} <: TensorMixed{V} <: TensorAlgebra{V}

Sparse multivector type with pseudoscalar `V::Manifold` and grade encoding `G::UInt64`.
""" MultiGrade

terms(v::MultiGrade) = v.v
value(v::MultiGrade) = reduce(vcat,value.(terms(v)))

MultiGrade{V}(v::Vector{T}) where T<:TensorGraded{V} where V = MultiGrade{V,|(UInt(1).<<rank.(v)...)}(Values(v...))
MultiGrade(v::Vector{T}) where T<:TensorGraded{V} where V = MultiGrade{V}(v)
MultiGrade(m::T) where T<:TensorAlgebra = m
MultiGrade(m::Chain{T,V,G}) where {T,V,G} = chainvalues(V,value(m),Val{G}())

function MultiGrade(m::MultiVector{V,T}) where {V,T}
    N = mdims(V)
    sum(m.v .== 0)/(1<<N) < fill_limit && (return m)
    out = zeros(FixedVector{N,TensorGraded{V}})
    G = zero(UInt)
    for i ∈ 0:N
        @inbounds !prod(m[i].==0) && (G|=UInt(1)<<i;out[i+1]=chainvalues(V,m[i],Val{i}()))
    end
    cG = count_ones(G)
    return cG≠1 ? MultiGrade{V,G}(Values{cG,T}(out[indices(G,N+1)]...)) : out[1]
end

function show(io::IO, m::MultiGrade{V,G}) where {V,G}
    t = terms(m)
    isempty(t) && print(io,zero(V))
    for k ∈ 1:count_ones(G)
        k ≠ 1 && print(io," + ")
        print(io,t[k])
    end
end

#=function MultiVector{V,T}(v::MultiGrade{V}) where {V,T}
    N = mdims(V)
    sigcheck(v.s,V)
    g = rank.(v.v)
    out = zeros(mvec(N,T))
    for k ∈ 1:length(v.v)
        @inbounds (val,b) = typeof(v.v[k]) <: Basis ? (one(T),v.v[k]) : (v.v[k].v,basis(v.v[k]))
        setmulti!(out,convert(T,val),bits(b),Val{N}())
    end
    return MultiVector{V}(out)
end

MultiVector{V}(v::MultiGrade{V}) where V = MultiVector{V}(v)
MultiVector(v::MultiGrade{V}) where V = MultiVector{V,promote_type(typeval.(v.v)...)}(v)=#

==(a::MultiGrade{V,G},b::MultiGrade{V,G}) where {V,G} = prod(terms(a) .== terms(b))

valuetype(t::MultiGrade) = promote_type(valuetype.(terms(t))...)
@inline value(m::MultiGrade,T) = m

@inline scalar(t::MultiGrade{V,G}) where {V,G} = @inbounds 1 ∈ indices(G) ? terms(t)[1] : zero(V)
@inline vector(t::MultiGrade{V,G}) where {V,G} = @inbounds (i=indices(G);2∈i ? terms(t)[findfirst(x->x==2,i)] : zero(V))
@inline volume(t::MultiGrade{V,G}) where {V,G} = @inbounds mdims(V)+1∈indices(G) ? terms(t)[end] : zero(V)
@inline isscalar(t::MultiGrade) = norm(t) ≈ scalar(t)
@inline isvector(t::MultiGrade) = norm(t) ≈ norm(vector(t))

adjoint(b::MultiGrade{V,G}) where {V,G} = MultiGrade{dual(V),G}(adjoint.(terms(b)))

## Generic

import Base: isinf, isapprox
import Leibniz: basis, grade, order
import AbstractTensors: value, valuetype, scalar, isscalar, involute, unit, even, odd
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume, ⋆
import LinearAlgebra: rank, norm
export basis, grade, hasinf, hasorigin, scalar, norm, gdims, betti, χ
export valuetype, scalar, isscalar, vector, isvector, indices

@pure valuetype(t::SparseChain{V,G,T} where {V,G}) where T = T

@inline value(m::SparseChain,T=valuetype(m)) = T∉(valuetype(m),Any) ? convert(SparseVector{T,Int},m.v) : m.v

@inline scalar(t::SparseChain{V,0}) where V = @inbounds Simplex{V}(t.v[1])

## Adjoint

import Base: adjoint # conj

adjoint(b::SparseChain{V,G}) where {V,G} = SparseChain{dual(V),G}(adjoint.(terms(b)))


