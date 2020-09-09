
#   This file is part of Clifford.jl. It is licensed under the AGPL license
#   Clifford Copyright (C) 2019 Michael Reed

for (op,eop) ∈ ((:+,:(+=)),(:-,:(-=)))
    @eval begin
        function $op(a::SparseChain{V,G,T},b::SparseChain{V,G,S}) where {V,G,T,S}
            isempty(a.v.nzval) && (return b)
            isempty(b.v.nzval) && (return a)
            t = length(a.v.nzind) > length(b.v.nzind)
            bi,bv = value(t ? b : a).nzind,value(t ? b : a).nzval
            out = convert(SparseVector{promote_type(T,S),Int},copy(value(t ? a : b)))
            $(Expr(eop,:(out[bi]),:bv))
            SparseChain{V,G}(out)
        end
        function $op(a::SparseChain{V,G,S},b::T) where T<:TensorTerm{V,G} where {V,G,S}
            out = convert(SparseVector{promote_type(S,valuetype(b)),Int},copy(value(a)))
            $(Expr(eop,:(out[basisindex(mdims(V),bits(b))]),:(value(b))))
            SparseChain{V,G}(out)
        end
    end
    for Tens ∈ (:(TensorTerm{V,B}),:(Chain{T,V,B} where T))
        @eval $op(a::T,b::SparseChain{V,A}) where {T<:$Tens} where {V,A,B} = b+a
    end
    @eval begin
        function $op(a::MultiGrade{V,A},b::MultiGrade{V,B}) where {V,A,B}
            at,bt = terms(a),terms(b)
            isempty(at) && (return b)
            isempty(bt) && (return a)
            bl = length(bt)
            out = convert(Vector{TensorGraded{V}},at)
            N = mdims(V)
            i,k,bk = 0,1,rank(out[1])
            while i < bl
                k += 1
                i += 1
                bas = rank(bt[i])
                if bas == bk
                    $(Expr(eop,:(out[k-1]),:(bt[i])))
                    k < length(out) ? (bk = rank(out[k])) : (k -= 1)
                elseif bas<bk
                    insert!(out,k-1,bt[i])
                elseif k ≤ length(out)
                    bk = rank(out[k])
                    i -= 1
                else
                    insert!(out,k,bt[i])
                end
            end
            G = A|B
            MultiGrade{V,G}(Values{count_ones(G),TensorGraded{V}}(out))
        end
        function $op(a::MultiGrade{V,A},b::T) where T<:TensorGraded{V,B} where {V,A,B}
            N = mdims(V)
            out = convert(Vector{TensorGraded{V}},terms(a))
            i,k,bk,bl = 0,1,rank(out[1]),length(out)
            while i < bl
                k += 1
                i += 1
                if bk == B
                    $(Expr(eop,:(out[k-1]),:b))
                    break
                elseif B<bk
                    insert!(out,k-1,b)
                    break
                elseif k ≤ length(out)
                    bk = rank(out[k])
                else
                    insert!(out,k,b)
                    break
                end
            end
            G = A|(UInt(1)<<B)
            MultiGrade{V,G}(Values{count_ones(G),TensorGraded{V}}(out))
        end
        $op(a::SparseChain{V,A},b::T) where T<:TensorGraded{V,B} where {V,A,B} = MultiGrade{V,(UInt(1)<<A)|(UInt(1)<<B)}(A<B ? Values(a,b) : Values(b,a))
    end
end


