`delsum`
========

`delsum` is a cli application for finding out checksums used in a file.

There are currently three subcommands:
* `check`: given a specification of the checksum algorithm and a list of files, this simply outputs the checksums of these files.
* `part`: given a specification of the checksum algorithm and a list of files with corresponding checksums, this finds parts of the files that have the given checksum
* `reverse`: given a list of files with corresponding checksums, this finds the checksum parameters used

`check`
-------
This subcommand calculates a checksum with a given algorithm.

Example:
```
$ delsum check -m 'crc width=32 poly=0x4c11db7 init=0xffffffff xorout=0xffffffff refin=true refout=true' file_a file_b file_c
700b14f5,e1207917,79741cb2
```
An algorithm can be specified directly as an argument with `-m` or a list of algorithms can be provided in a file.

The start or end of the range to calculate the checksum of is given with `-S` or `-E` and can also be negative to make them relative to the end.
For example, to calculate the checksum of all bytes except the last two, one specifies `-S 0` and `-E-3` (it is inclusive, so `-3` is still part of the sum).

For the available algorithms and how to specify them, see [here](#algorithms).

`part`
------
This subcommand finds all parts of a list of files where all given checksums match.
The parts of the file that match has to be the same accross the files.

Example:
```
$ delsum part -m 'modsum width=16 modulus=0xffff' -c 1234,5678,abcd file_a file_b file_c
modsum width=16 modulus=0xffff:
    0x8:-0x3
```

In this case, the checksum matches in the part `0x8:-0x3`, where `-0x3` is relative from the end.
This is an inclusive range, meaning that a checksum that goes from the start to the end would have the range `0x0:-0x1`.
If the files were of sizes 15, 16 and 18 bytes respectively, then this output would mean that
* `file_a` has checksum `1234` from byte 8 to 13
* `file_b` has checksum `5678` from byte 8 to 14
* `file_c` has checksum `abcd` from byte 8 to 16

One can also have the end of parts be relative from the start of the file (and not the end) by using the `-s` flag.
Furthermore, the `-S` and `-E` allow one to constrain where ranges begin and end by specifying ranges (also inclusive).
For example, when normally, the ranges `0x1:0xa`, `0x3:0x10` and `0x4:0xb` would be output, specifying `-S0x0:0x3` would only allow the start part of the ranges to be between 0 and 3 inclusive, so `0x4` would not be printed.
This can help avoid false positives and can also reduce execution time.

There's a small chance that it will output something like `0x1,0x6:0x5,0x10` is output.
This just means that each combination is possible.
In this case, one would have `0x1:0x5`, `0x1:0x10` and `0x6:0x10`.
While `0x6:0x5` would theoretically also be a choice, it is not a valid one since it is backwards.

By exploiting the linearity of the checksums, this whole process can be done in roughly loglinear time, but just keep in mind that
it has a big (linear) space overhead and you might run out of memory if you run it on a bunch of 500MB files.

One can also give a list of algorithms in a file as an input to `-M`.
This can be useful, as it allows to simply put the most common few checksum algorithm in there and look if any algorithms in any part of the files has the desired checksum.
For the available algorithms and how to specify them, see [here](#algorithms).

`reverse`
---------
This subcommand finds parameters of a checksum algorithm.

With given files and checksums, it searches those algorithm parameters for which the entire files have the given checksums.

Note that at least the fundamental algorithm and the width of the checksum must be specified (for example, `crc width=32`).
For the available algorithms and how to specify them, see [here](#algorithms).

Example:
```
$ delsum reverse -m 'crc width=32' -c 700b14f5,e1207917,79741cb2 file_a file_b file_c
crc width=32 poly=0x4c11db7 init=0xffffffff xorout=0xffffffff refin=true refout=true
```

You generally need 3 files, and for algorithms other than `modsum` at least one of the files needs to have a different length.
It is also possible to specify some parameters of the algorithm (using for example `-m 'crc width=32 init=0'`), which needs fewer files or yields fewer false positives.

If you have only files of a given length, but also only care about checksums of that length, for an algorithm not `modsum` you can simply set `init=0`.

It is normally quite fast; for example the runtime for the CRC reversing algorithm is in most cases around `O(n*log^2(n)*log(log(n)))` where `n` is the filesize, which is thanks to the fast multiplication algorithm implemented within the gf2x library.

For some parameters, only likely combinations are searched:
* `wordsize` is searched for powers of 2 multiples of 8 that are smaller or equal to `width`
* `refin` and `refout` is searched for `refin = refout`

To search these parameters, either specify them manually or use the `--extended-search` cli argument.

Algorithms
----------
There are currently three families of algorithms: `modsum`, `fletcher` and `crc`.
They are specified like this: `algofamiliy width=123 para1=ff para2=true para3=10 name="algoname"`.
Note that all numerical parameters except `width` and `wordsize` are in hexadecimal.

Common Values
=============
Currently, these are shared accross all sum types:
* `width`: Width of the checksum in bits, decimal.
* `out_endian`: Endian of the checksum, can be either `little` or `big`, defaults to `big`.
* `wordsize`: Number of bits of a word in the input text, decimal.
              Must be a multiple of 8 and between 8 and 64.
              For example, in a simple checksum, using `wordsize=16` would chop the file in into 16-bit integers and add them up modulo `modulus`.
* `in_endian`: the endian of the input words, can be either `little` or `big`, defaults to `big`.
* `signedness`: the signedness of the input words, can be either `signed` (using 2's complement) or `unsigned`. Not valid for `crc`.

`modsum`
========
A simple modular sum with parameters `width`, `init`, `modulus` and `negated`.

Corresponds to
```
sum = init
for byte in file:
    sum = (sum + byte) % modulus
return -sum if negated else sum
```
The parameters are:
* `width`: The width of the checksum. Mandatory.
* `modulus`: The value by which to reduce, hexadecimal. `modulus=0x00` means `2^width` and is the default value.
* `init`: The value to initialize the regular checksum with, hexadecimal. Defaults to 0.
* `negated`: The boolean flag which indicates that the checksum should be negated, `true`/`false`. Defaults to `false`.

`fletcher`
==========
A fletcher-like sum with parameters `width`, `init`, `addout`, `modulus` and `swap`.

Corresponds to
```
sum1 = init
sum2 = 0
for byte in file:
    sum1 = (sum1 + byte) % modulus
    sum2 = (sum2 + sum1) % modulus
sum1 = (sum1 + addout.sum1) % modulus
sum2 = (sum2 + addout.sum2) % modulus
if not swap:
    returm (sum2 << (width/2)) | sum1
else:
    returm (sum1 << (width/2)) | sum2
```

It is output in a "packed" form where the sum1 is stored in the lower width/2 bits and sum2 in the higher width/2 (or the opposite if `swap` is enabled).
The parameters are:
* `width`: The width of the whole packed checksum. Mandatory.
* `modulus`: The value by which to reduce, hexadecimal. `modulus=0x00` means `2^(width/2)` and is the default value.
* `init`: The value to initialize the regular checksum with, hexadecimal. Defaults to 0.
* `addout`: The packed value which is added at the end of the sum, hexadecimal. The high part is always added to the high part of the checksum at the end, regardless of `swap`. Defaults to 0.
* `swap`: The boolean flag which indicates that the regular sum should be in the higher half of the packed checksum, `true`/`false`. Defaults to `false`.

`crc`
=====
A CRC algorithm with parameters in accordance to the Rocksoft^TM model, as documented in "A Painless Guide to Crc Error Detection".

It has the following parameters:
* `width`: The width in bits of the checksum (and degree of poly). Mandatory.
* `poly`: The generator polynomial, in normal notation, hexadecimal. Mandatory (except for `reverse`).
* `init`: The initial value of the crc state, hexadecimal. Defaults to 0.
* `xorout`: The final value to xor to the sum, hexadecimal. Defaults to 0.
* `refin`: The boolean flag indicating whether to reflect the bits of the input bytes, `true`/`false`. Defaults to `false`.
* `refout`: The boolean flag indicating whether to reflect the bits of the final checksum, before adding `xorout`, `true`/`false`. Defaults to `false`.

Note that other values for `wordsize` with `in_endian=little` (the standard) is the same as swapping the bytes in each group of `wordsize` bits before calculating the `wordsize=8` checksum.

`polyhash`
==========
A polynomial hash function with parameters `width`, `factor`, `init` and `addout`.

Corresponds to
```
sum = init
for byte in file:
    sum = (sum * factor + byte) % 2**width
return (sum + addout) % 2**width
```

The parameters are:
* `width`: The width of the checksum. Mandatory.
* `factor`: The factor to multiply with. Mandatory (except for `reverse`).
* `init`: The value to initialize the regular checksum with. Defaults to 0.
* `addout`: The value to add at the end. Defaults to 0.

For example, the djb2 hash would be `polyhash width=32 factor=0x21 init=0x1505`.

How this works
--------------
Some (incomplete) explanation of the algorithms used is found [here](algorithms.md).

Installing
----------
There is a linux build which has the gf2x library compiled in [here](https://github.com/8051Enthusiast/delsum/releases), but keep in mind that it is compiled without most modern x86 extensions and therefore can't take advantage of some optimized routines in `gf2x` which makes CRC reversing a lot faster.
I'm also too dumb for doing a Windows build, so sorry for that.

This program links against the [`gf2x`](https://gitlab.inria.fr/gf2x/gf2x) library.

If you're on a Debian-based system, you can install it with
```
apt-get install libgf2x-dev
```

You can also compile them yourselves, see [here](https://gitlab.inria.fr/gf2x/gf2x). This will generally make the fastest binary,
as instruction set extensions can be used and there is also the possible of tuning the algorithm parameters.

If you have `cargo` installed, it should then be possible to compile this in the project directory root with
```
cargo install --path .
```
or, without downloading the repository, with
```
cargo install delsum
```

If you want to link the gf2x library statically, you can set the environment variable `GF2POLY_STATIC_LIB=1` when running `cargo`.

Acknowledgements
----------------
The fast crc reversing algorithm would not be fast without the work of the authors of the `gf2x` library.
This project also previously used the `NTL` library for fast gcd and factorization, before the author wrote their own library for it.

A big thanks also to the authors of [`CRC RevEng`](https://reveng.sourceforge.io/) for cataloging the CRC algorithms and their parameters, and prior work on the CRC reversing algorithm.

License
-------
The code of this project is licensed under the MIT license.
