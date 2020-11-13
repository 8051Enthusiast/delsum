#include "delsum-poly/include/poly.hh"
#include "delsum-poly/src/lib.rs.h"
#include <cstdint>
#include <climits>
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
namespace poly
{
    long i64_to_l(int64_t x) {
        if (x > LONG_MAX || x < LONG_MIN) {
            std::cerr << "Polynomial too big, probably because of big input files" << std::endl
                      << "Try using smaller files or run on a platform with larger long variables" << std::endl;
            abort();
        }
        return (long)x;
    }
    Poly::Poly()
    {
        int_pol = NTL::GF2X();
    }
    Poly::Poly(const Poly &b)
    {
        int_pol = NTL::GF2X(b.int_pol);
    }
    Poly::Poly(Poly &&b)
    {
        int_pol = NTL::GF2X(b.int_pol);
    }
    bool Poly::coeff(int64_t idx) const
    {
        return NTL::IsOne(NTL::coeff(int_pol, i64_to_l(idx)));
    }
    bool Poly::eq(const Poly &b) const
    {
        return int_pol == b.int_pol;
    }
    int64_t deg(const Poly &a)
    {
        return NTL::deg(a.int_pol);
    }
    std::unique_ptr<Poly> copy_poly(const Poly &p)
    {
        return std::make_unique<Poly>(p);
    }
    std::unique_ptr<std::vector<PolyI64Pair>> factor(const Poly &p, int64_t verbosity)
    {
        auto v = std::vector<PolyI64Pair>();
        auto decomp = NTL::CanZass(p.int_pol, i64_to_l(verbosity));
        for (auto x : decomp)
        {
            auto poly = Poly();
            poly.int_pol = x.a;
            v.push_back(PolyI64Pair{.poly = std::make_unique<Poly>(std::move(poly)), .l = (int64_t)x.b});
        }
        return std::make_unique<std::vector<PolyI64Pair>>(std::move(v));
    }

    std::unique_ptr<Poly> new_poly_shifted(rust::Slice<uint8_t> bytes, int64_t shift, bool msb_first)
    {
        auto lshift = i64_to_l(shift);
        auto ret = Poly();
        ret.int_pol = NTL::GF2X();
        ret.int_pol.SetLength(lshift + bytes.length() * 8);
        for (size_t i = 0; i < bytes.length(); i++)
        {
            auto current_byte = bytes.data()[bytes.length() - 1 - i];
            for (int j = 0; j < 8; j++)
            {
                auto bit_pos = msb_first ? j : (7 - j);
                auto current_bit = (current_byte >> bit_pos) & 1;
                auto bit_index = lshift + 8 * i + j;
                ret.int_pol[bit_index] = current_bit;
            }
        }
        ret.int_pol.normalize();
        return std::make_unique<Poly>(ret);
    }

    std::unique_ptr<Poly> new_poly(rust::Slice<uint8_t> bytes)
    {
        return new_poly_shifted(bytes, 0, true);
    }
    std::unique_ptr<Poly> new_zero()
    {
        auto ret = Poly();
        return std::make_unique<Poly>(ret);
    }

    std::unique_ptr<std::vector<uint8_t>> Poly::to_bytes(int64_t min_bytes) const
    {
        auto d = NTL::deg(int_pol);
        auto n_bytes = d / 8 + 1;
        if (d < 0) {
            n_bytes = 0;
        }
        auto v = std::vector<uint8_t>();
        auto lmin = i64_to_l(min_bytes);
        auto amount_of_bytes = n_bytes > lmin ? n_bytes : lmin;
        v.reserve(amount_of_bytes);
        for (long i = 0; i < amount_of_bytes - n_bytes; i++) {
            v.push_back(0);
        }
        uint8_t current_byte = 0;
        for (long i = d; i >= 0; i--)
        {
            current_byte <<= 1;
            current_byte |= (uint8_t)NTL::rep(int_pol[i]);
            if (i % 8 == 0)
            {
                v.push_back(current_byte);
                current_byte = 0;
            }
        }
        return std::make_unique<std::vector<uint8_t>>(v);
    }
    std::unique_ptr<Poly> add(const Poly &b, const Poly &c)
    {
        auto ret = Poly();
        ret.int_pol = b.int_pol + c.int_pol;
        return std::make_unique<Poly>(ret);
    }
    void Poly::add_to(const Poly &b)
    {
        int_pol += b.int_pol;
    }
    std::unique_ptr<Poly> mul(const Poly &b, const Poly &c)
    {
        auto ret = Poly();
        ret.int_pol = b.int_pol * c.int_pol;
        return std::make_unique<Poly>(ret);
    }
    void Poly::mul_to(const Poly &b)
    {
        int_pol *= b.int_pol;
    }
    std::unique_ptr<Poly> div(const Poly &b, const Poly &c)
    {
        auto ret = Poly();
        ret.int_pol = b.int_pol / c.int_pol;
        return std::make_unique<Poly>(ret);
    }
    void Poly::div_to(const Poly &b)
    {
        int_pol /= b.int_pol;
    }
    bool Poly::div_to_checked(const Poly &b)
    {
        auto ret = NTL::divide(int_pol, b.int_pol);
        return ret == 1;
    }
    bool Poly::is_zero() const
    {
        return NTL::IsZero(int_pol);
    }
    std::unique_ptr<Poly> gcd(const Poly &b, const Poly &c)
    {
        auto ret = Poly();
        NTL::GCD(ret.int_pol, b.int_pol, c.int_pol);
        return std::make_unique<Poly>(ret);
    }
    std::unique_ptr<Poly> xgcd(Poly &x, Poly &y, const Poly &b, const Poly &c)
    {
        auto ret = Poly();
        NTL::XGCD(ret.int_pol, x.int_pol, y.int_pol, b.int_pol, c.int_pol);
        return std::make_unique<Poly>(ret);
    }
    void Poly::gcd_to(const Poly &b)
    {
        NTL::GCD(int_pol, int_pol, b.int_pol);
    }
    std::unique_ptr<Poly> rem(const Poly &b, const Poly &c)
    {
        auto ret = Poly();
        NTL::rem(ret.int_pol, b.int_pol, c.int_pol);
        return std::make_unique<Poly>(ret);
    }
    void Poly::rem_to(const Poly &b)
    {
        NTL::rem(int_pol, int_pol, b.int_pol);
    }
    std::unique_ptr<Poly> power(const Poly &p, int64_t n)
    {
        auto q = Poly();
        q.int_pol = NTL::power(p.int_pol, i64_to_l(n));
        return std::make_unique<Poly>(q);
    }
    std::unique_ptr<Poly> shift(const Poly &p, int64_t n)
    {
        auto q = Poly();
        if (n >= 0) {
            q.int_pol = NTL::LeftShift(p.int_pol, i64_to_l(n));
        } else {
            q.int_pol = NTL::RightShift(p.int_pol, i64_to_l(-n));
        }
        return std::make_unique<Poly>(q);
    }
    void Poly::sqr()
    {
        int_pol = NTL::sqr(int_pol);
    }

    // PolyRem stuff

    PolyRem::PolyRem(const Poly &p)
    {
        int_pol = NTL::GF2E();
        int_pol.init(p.int_pol);
    }
    // copy
    PolyRem::PolyRem(const PolyRem &b)
    {
        int_pol = NTL::GF2E(b.int_pol);
    }
    // move
    PolyRem::PolyRem(PolyRem &&b)
    {
        int_pol = NTL::GF2E(b.int_pol);
    }
    PolyRem::PolyRem(const NTL::GF2E &p)
    {
        int_pol = NTL::GF2E(p);
    }

    std::unique_ptr<PolyRem> new_polyrem(const Poly &rem, const Poly &m)
    {
        auto ret = PolyRem(m);
        NTL::conv(ret.int_pol, rem.int_pol);
        return std::make_unique<PolyRem>(ret);
    }
    void PolyRem::add_to(const PolyRem &b)
    {
        int_pol += b.int_pol;
    }
    void PolyRem::mul_to(const PolyRem &b)
    {
        int_pol *= b.int_pol;
    }
    void PolyRem::div_to(const PolyRem &b)
    {
        int_pol /= b.int_pol;
    }
    void PolyRem::sqr()
    {
        int_pol = NTL::sqr(int_pol);
    }
    std::unique_ptr<Poly> PolyRem::rep() const
    {
        auto ret = Poly();
        ret.int_pol = NTL::rep(int_pol);
        return std::make_unique<Poly>(ret);
    }
    std::unique_ptr<PolyRem> powermod(const PolyRem &p, int64_t n)
    {
        auto q = PolyRem(NTL::power(p.int_pol, i64_to_l(n)));
        return std::make_unique<PolyRem>(q);
    }
    std::unique_ptr<PolyRem> copy_polyrem(const PolyRem &p)
    {
        return std::make_unique<PolyRem>(PolyRem(p));
    }
} // namespace poly
