#pragma once
#include "rust/cxx.h"
#include <memory>
#include <cstdint>
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>

namespace poly
{
    struct PolyI64Pair;
    class Poly
    {
    public:
        // internal representation of polynomial
        NTL::GF2X int_pol;
        Poly();
        // copy
        Poly(const Poly &b);
        // move
        Poly(Poly &&b);
        void add_to(const Poly &b);
        void mul_to(const Poly &b);
        void div_to(const Poly &b);
        void gcd_to(const Poly &b);
        void rem_to(const Poly &b);
        void sqr();
        bool div_to_checked(const Poly &b);
        bool coeff(int64_t idx) const;
        bool eq(const Poly &b) const;
        bool is_zero() const;
        std::unique_ptr<std::vector<uint8_t>> to_bytes(int64_t min_bytes) const;
    };

    int64_t deg(const Poly &a);
    std::unique_ptr<Poly> new_poly_shifted(rust::Slice<const uint8_t> bytes, int64_t shift, bool msb_first);
    std::unique_ptr<Poly> new_poly(rust::Slice<const uint8_t> bytes);
    std::unique_ptr<Poly> new_zero();
    std::unique_ptr<Poly> copy_poly(const Poly &p);
    std::unique_ptr<Poly> add(const Poly &b, const Poly &c);
    std::unique_ptr<Poly> mul(const Poly &b, const Poly &c);
    std::unique_ptr<Poly> div(const Poly &b, const Poly &c);
    std::unique_ptr<Poly> gcd(const Poly &b, const Poly &c);
    std::unique_ptr<Poly> xgcd(Poly &x, Poly &y, const Poly &b, const Poly &c);
    std::unique_ptr<Poly> rem(const Poly &b, const Poly &c);
    std::unique_ptr<Poly> power(const Poly &p, int64_t n);
    std::unique_ptr<Poly> shift(const Poly &p, int64_t n);
    std::unique_ptr<std::vector<PolyI64Pair>> factor(const Poly &p, int64_t verbosity);

    class PolyRem
    {
    public:
        NTL::GF2E int_pol;
        // from modulus
        PolyRem(const Poly &p);
        // copy
        PolyRem(const PolyRem &b);
        // move
        PolyRem(PolyRem &&b);
        PolyRem(const NTL::GF2E &p);
        void add_to(const PolyRem &b);
        void mul_to(const PolyRem &b);
        void div_to(const PolyRem &b);
        void sqr();
        std::unique_ptr<Poly> rep() const;
    };
    std::unique_ptr<PolyRem> new_polyrem(const Poly &rem, const Poly &m);
    std::unique_ptr<PolyRem> powermod(const PolyRem &p, int64_t n);
    std::unique_ptr<PolyRem> copy_polyrem(const PolyRem &p);
} // namespace poly
