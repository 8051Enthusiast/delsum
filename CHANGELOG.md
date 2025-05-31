1.0.0 (2025-05-31)
==================
* Change the `module` parameter to be named `modulus`
* Change all hexadecimal parameters to now need a `0x` prefix
* Add new `polyhash` algorithm kind
* Add JSON output
* Add `-t` option for trailing checksum bytes that directly follow the content
* Add some scripts using delsum (inside `/scripts`)
* Add `signedness` parameter to `modsum`/`fletcher`/`polyhash`
* Add `negate` parameter to `modsum`
* Add website
* Improve error messages somewhat
* Fix various behaviours when the checksum given by the user is bigger
  than the range the algorithm can output
* Fix `part` not outputting ranges with a length of 1 byte in most cases
* Remove NTL from dependencies

0.2.1 (2024-01-28)
==================
* Remove gmp from dependencies
* Update dependencies

0.2.0 (2021-06-06)
==================
* introduce wordsize, out_endian and in_endian parameters to allow for checksums with words bigger than 8 bits
* minor API changes for delsum-lib
* `part` now uses inclusive ranges instead of right-exclusive ones
* all modes now support address ranges to operate on in files, for `part`, the start and end of the ranges
  can be ranges themselves
* make `reverse` only output new errors
* don't try very unlikely parameter combinations on default for `reverse`

0.1.2 (2020-11-24)
==================
* bump cxx to 1.0.2
* fix bug in factorization code (relevant for modsum and fletcher reversing)
* add changelog

0.1.0 (2020-11-14)
==================
Initial release
