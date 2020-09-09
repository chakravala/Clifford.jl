
#   This file is part of Clifford.jl. It is licensed under the AGPL license
#   Clifford Copyright (C) 2019 Michael Reed

for un ∈ (:complementleft,:complementright)
    @eval begin
        $un(t::SparseChain{V,G}) where {V,G} = SparseChain{V,mdims(V)-G}($un.(terms(t)))
        $un(t::MultiGrade{V,G}) where {V,G} = SparseChain{V,G⊻(UInt(1)<<mdims(V)-1)}(reverse($un.(terms(t))))
    end
end
for un ∈ (:reverse,:involute,:conj,:+,:-)
    @eval begin
        $un(t::SparseChain{V,G}) where {V,G} = SparseChain{V,G}($un.(terms(t)))
        $un(t::MultiGrade{V,G}) where {V,G} = SparseChain{V,G}($un.(terms(t)))
    end
end

function generate_sums(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj,PAR=false)
    if Field == Grassmann.Field
        generate_mutators(:(Variables{M,T}),Number,Expr,SUB,MUL)
    elseif Field ∈ (SymField,:(SymPy.Sym))
        generate_mutators(:(FixedVector{M,T}),Field,set_val,SUB,MUL)
    end
    PAR && (DirectSum.extend_field(eval(Field)); global parsym = (parsym...,eval(Field)))
    TF = Field ∉ FieldsBig ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    @eval begin
        *(a::F,b::MultiGrade{V,G}) where {F<:$EF,V,G} = MultiGrade{V,G}(broadcast($MUL,Ref(a),b.v))
        *(a::MultiGrade{V,G},b::F) where {F<:$EF,V,G} = MultiGrade{V,G}(broadcast($MUL,a.v,Ref(b)))
    end
    for (op,eop,bop) ∈ ((:+,:(+=),ADD),(:-,:(-=),SUB))
        @eval begin
            function $op(a::SparseChain{V,G,S},b::MultiVector{V,T}) where {V,T<:$Field,G,S<:$Field}
                $(insert_expr((:N,:t),VEC)...)
                at = value(a)
                out = convert($VEC(N,t),$(bcast(bop,:(copy(value(b,$VEC(N,t))),))))
                addmulti!(out,at.nzval,binomsum(N,G).+at.nzind)
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::SparseChain{V,G}) where {V,T<:$Field,G}
                $(insert_expr((:N,:t),VEC)...)
                bt = value(b)
                out = copy(value(a,$VEC(N,t)))
                addmulti!(out,$bop.(bt.nzval),binomsum(N,G).+bt.nzind)
                return MultiVector{V}(out)
            end
            function $op(a::Chain{V,G,T},b::SparseChain{V,G}) where {V,G,T<:$Field}
                $(insert_expr((:N,),VEC)...)
                bt = terms(b)
                t = promote_type(T,valuetype.(bt)...)
                out = copy(value(a,$VEC(N,G,t)))
                addmulti!(out,bt.nzval,bt.nzind)
                return Chain{V,G}(out)
            end
            function $op(a::SparseChain{V,G},b::Chain{V,G,T}) where {V,G,T<:$Field}
                $(insert_expr((:N,),VEC)...)
                at = terms(a)
                t = promote_type(T,valuetype.(at)...)
                out = convert($VEC(N,G,t),$(bcast(bop,:(copy(value(b,$VEC(N,G,t))),))))
                addmulti!(out,at.nzval,at.nzind)
                return Chain{V,G}(out)
            end
            function $op(a::MultiGrade{V,G},b::MultiVector{V,T}) where {V,T<:$Field,G}
                $(insert_expr((:N,),VEC)...)
                at = terms(a)
                t = promote_type(T,valuetype.(at)...)
                out = convert($VEC(N,t),$(bcast(bop,:(value(b,$VEC(N,t)),))))
                for A ∈ at
                    TA = typeof(A)
                    if TA <: TensorTerm
                        addmulti!(out,value(A,t),bits(A),Val{N}())
                    elseif TA <: SparseChain
                        vA = value(A,t)
                        addmulti!(out,vA.nzval,vA.nzind)
                    else
                        g = rank(A)
                        r = binomsum(N,g)
                        @inbounds $(add_val(eop,:(out[r+1:r+binomial(N,g)]),:(value(A,$VEC(N,g,t))),bop))
                    end
                end
                return MultiVector{V}(out)
            end
            function $op(a::MultiVector{V,T},b::MultiGrade{V,G}) where {V,T<:$Field,G}
                $(insert_expr((:N,),VEC)...)
                bt = terms(b)
                t = promote_type(T,valuetype.(bt)...)
                out = value(a,$VEC(N,t))
                for B ∈ bt
                    TB = typeof(B)
                    if TB <: TensorTerm
                        addmulti!(out,$bop(value(B,t)),bits(B),Val{N}())
                    elseif TB <: SparseChain
                        vB = value(B,t)
                        addmulti!(out,vB.nzval,vB.nzind)
                    else
                        g = rank(B)
                        r = binomsum(N,g)
                        @inbounds $(add_val(eop,:(out[r+1:r+binomial(N,g)]),:(value(B,$VEC(N,g,t))),bop))
                    end
                end
                return MultiVector{V}(out)
            end
        end
    end
end

### Product Algebra Constructor

insert_t(x) = Expr(:block,:(t=promote_type(valuetype(a),valuetype(b))),x)

function generate_products(Field=Field,VEC=:mvec,MUL=:*,ADD=:+,SUB=:-,CONJ=:conj,PAR=false)
    TF = Field ∉ FieldsBig ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    generate_sums(Field,VEC,MUL,ADD,SUB,CONJ,PAR)
    #=@eval begin
        ∧(a::$Field,b::MultiGrade{V,G}) where V = MultiGrade{V,G}(a.*b.v)
        ∧(a::MultiGrade{V,G},b::$Field) where V = MultiGrade{V,G}(a.v.*b)
    end=#
end
