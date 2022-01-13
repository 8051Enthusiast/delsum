Algorithms for reverse engineerign CRCs
=============================================================

𝔽<sub>2</sub>[X]
----------------

When working with CRCs, we work in the ring of polynomials over the field 𝔽<sub>2</sub> (mostly modulo some polynomial).
Polynomials are typically notated like x<sup>5</sup> + x<sup>2</sup> + x<sup>1</sup> + 1 but I will also notate them like 100111 (for each exponent, just the coefficient is written) in this document.
Addition is then done using XOR and multiplication is carry-less multiplication, which works like normal multiplication but in each step of the calculation, one uses XOR instead of normal addition.
For example, `1010 + 1100 = 0110` and
```
110010 · 1011 = 111000110
-------------
        1011
+    1011
+   1011
-------------
    111000110
```

From a number theoretical perspective, they are similar to the integers:
* the result of multiplicating two non-zero elements is always non-zero and multiplying by 1 leaves it the same.
* division with remainder is possible for each pair: the remainder then always has a degree smaller than the divisor. For example, 1000110100 (degree 9) modulo 11001 (degree 4) is 1101 (degree 3).
* two elements have a unique greatest common divisor that divides both elements. "Greatest" here means that it has the greatest degree.
* each element has a unique factorization. For example, 10001 may be written as 11<sup>4</sup> and 1000110100 as 10<sup>2</sup> · 11<sup>2</sup> · 101001.

One difference to integers is that anything multiplied by 1 + 1 (= 2) is zero because a number XORed by itself is always zero.
Because 1 + 1 = 0, one cannot make sign errors as 1 + 1 = 0 = 1 - 1 and therefore 1 = -1.

CRCs
----
There are some well-established parameters for a CRC algorithm:
* `init`: the initial state of the CRC register
* `poly`: the polynomial by which the sum gets reduced
* `width`: the degree of poly
* `xorout`: a final polynomial that gets added after everything has been done
* `refin`: whether to reflect the bits of the input bytes
* `refout`: whether to reflect the bits of the final sum

Let `f` be the input file represented as a polynomial (e.g. the bytes `0x45 0x11` would get represented as `01000101 00010001` or x<sup>14</sup> + x<sup>10</sup> + x<sup>8</sup> + x<sup>4</sup> + 1) and `len(f)` the file length in bits (which is not neccessarily the degree, as the high bits can be zero).
`refin` and `refout` will be ignored for now.

The CRC can then be represented as: `crc` = (`init` · x<sup>`len(f)`</sup> + `f` · x<sup>`width`</sup> + `xorout`) % `poly`.

Reverse Engineering the Algorithm Parameters from Files and Their Corresponding Checksums
-----------------------------------------------------------------------------------------
Suppose you have some files `f`<sub>k</sub> with k ranging from 1 to n.
Additionally, you have their corresponding checksums `crc`<sub>k</sub> using some unknown CRC algorithm.
Now you want to find the parameters for the algorithm.

### Finding `poly`
First off, we assume that we already know the width of the CRC (otherwise, looping over it is usually not that hard), which is everything needed to calculate

`a`<sub>k</sub> := (`crc`<sub>k</sub> - `f`<sub>k</sub> · x<sup>`width`</sup>) % `poly` \
&nbsp; = (`init` · x<sup>`len(f_k)`</sup> + `f`<sub>k</sub> · x<sup>`width`</sup> + `xorout` - `f`<sub>k</sub> · x<sup>`width`</sup>) % `poly` \
&nbsp; = (`init` · x<sup>`len(f_k)`</sup> + `xorout`) % `poly`

The resulting term does not depend on the file content, only its length.
Let's call that term `a`<sub>k</sub>.
If we calculate that term for each k, we can subtract pairs of consecutive terms and get

&nbsp; (`a`<sub>k+1</sub> - `a`<sub>k</sub>) % `poly` \
= (`init` · x<sup>`len(f_(k+1))`</sup> + `xorout` - `init` · x<sup>`len(f_k)`</sup> - `xorout`) % `poly` \
= `init` · (x<sup>`len(f_(k+1))`</sup> - x<sup>`len(f_k)`</sup>) % `poly`

Now it might be the case that `len(f_k)` = `len(f_(k+1))`  which would mean that the term modulo `poly` is already zero.
Since the remainder mod `poly` would be zero, `poly` would be a divisor of `a`<sub>k + 1</sub> - `a`<sub>k</sub> which gives us some information on the possible values of `poly`.
Indeed, we can calculate the greatest common divisor of all such result in order to combine all the information we get from that into a single value (let's call it `m`).
This is because each of those differences individually must be divisible by `poly`, therefore `poly` must be a divisor common to all of those values.

Additionally, we can still do something when the files are of different lengths.
If we have three files with `len(f_k)` < `len(f_(k+1))` < `len(f_(k+2))` (for clearer notation, call those lengths `l0`, `l1` and `l2`), we can cancel out the `init` term by noting the following equality:

&nbsp; (x<sup>`l2`</sup> - x<sup>`l1`</sup>) · (`a`<sub>k+1</sub> - `a`<sub>k</sub>) % `poly` \
= (x<sup>`l2`</sup> - x<sup>`l1`</sup>) · `init` · (x<sup>`l1`</sup> - x<sup>`l0`</sup>) % `poly` \
= (x<sup>`l1`</sup> - x<sup>`l0`</sup>) · (`a`<sub>k+2</sub> - `a`<sub>k+1</sub>) % `poly`

Which means that 

&nbsp; ((x<sup>`l2`</sup> - x<sup>`l1`</sup>) · (`a`<sub>k+1</sub> - `a`<sub>k</sub>) - (x<sup>`l1`</sup> - x<sup>`l0`</sup>) · (`a`<sub>k+2</sub> - `a`<sub>k+1</sub>)) % `poly` = 0

We can calculate the value on the left side of the modulo directly ourselves, which means we have a way of calculating a value that `poly` divides even if all files are of different sizes. That can be done for all consecutive triples of files (in order) and the values can all be gcd'ed together again with `m`.

Finally, we know that `poly` is of degree `width`, which means that a prime factor of degree `d` can occur at most `width` / `d` times. The product of all those prime factors is

&nbsp; `q` := ∏<sub>n=0..`width`</sub> (x<sup>2<sup>n</sup></sup> - x)

Why that expression has that property is not important in this context (the subexpression x<sup>2<sup>n</sup></sup> - x contains exactly those prime factors whose degree divides n).

One can see that the result of that expression would be quite large, however what we desire is gcd(`q`, `m`), which is the same as gcd(`q` % `m`, `m`) and the calculation modulo `m` should therefore not use much more space than `m` itself.

We can also remove any multiple of x from `m`, since `poly` will have a non-zero constant term, otherwise it would not be used in a CRC.

### Solving for `init`

Now that we have a non-zero upper bound (in the divisibility sense) on `poly` with `m`, we can find out `init` modulo `m`.
First, observe again what we get when we calculate a<sub>k + 1</sub> - a<sub>k</sub>:

&nbsp; a<sub> k + 1</sub> - a<sub>k</sub> = `init` · (x<sup>`len(f_(k+1))`</sup> - x<sup>`len(f_k)`</sup>) % `poly`

We do not have `poly`, but we can instead use `m`, which will yield a `init`' modulo m, and for each solution candidate `poly`, we can then simply calculate `init` = `init`' % `poly`.

So we now have a set of modular equations

for all k from 1 to n - 1:\
&nbsp; `init` · z<sub>k</sub> ≡ a<sub>k + 1</sub> - a<sub>k</sub> mod `m`\
where \
&nbsp; z<sub>k</sub> = x<sup>`len(f_(k+1))`</sup> - x<sup>`len(f_k)`</sup>\
&nbsp; y<sub>k</sub> = a<sub>k + 1</sub> - a<sub>k</sub>

The only value we do not have is `init`.
We could transform the equation into

&nbsp; `init` ≡ z<sub>k</sub><sup>-1</sup> · y<sub>k</sub> mod `m`

and directly get `init`, but wait!
z<sub>k</sub><sup>-1</sup> may not be invertible!

We can still get all the available information from each equation even if z<sub>k</sub> is not invertible:
first, y<sub>k</sub> and z<sub>k</sub> may have a common factor c<sub>k</sub> = gcd(y<sub>k</sub>, z<sub>k</sub>, `m`) which we can remove from z<sub>k</sub>, obtaining z' and y'

&nbsp; `init` · c<sub>k</sub> · z'<sub>k</sub> = c<sub>k</sub> · y'<sub>k</sub> mod `m`\
⇔ `init` · z'<sub>k</sub> = y'<sub>k</sub> mod `m` / c<sub>k</sub>

If z'<sub>k</sub> is still not invertible, there would be a non-one `m`' dividing `m` / c<sub>k</sub> such that z'<sub>k</sub> ≡ 0 mod `m`', but that would mean

&nbsp; `init` · z'<sub>k</sub> ≡ 0 ≡ y'<sub>k</sub> mod `m`'

which would imply that gcd(y'<sub>k</sub>, z'<sub>k</sub>, `m`') is not 1, contradicted by the fact that we factored c<sub>k</sub> out.

Since we assume that the equation modulo `poly` *does* have a solution, we remove those problematic factors from `m` until we get something solvable again (when we reach something with degree smaller than `width`, we can always abort and output that no solutions were found).

For each equation, there is now a solution of `init` mod `m` / c<sub>k</sub> with `m` having all problematic factors from each equation removed.
We can then combine all these solutions with (almost) the chinese remainder theorem.

The traditional chinese remainder theorem assumes that all modulos are coprime, which is not our case.
Instead, in our simplified chinese remainder theorem we make sure that the solutions are the same on the overlapping factors.
If they are not the same, the factors where the solution differs gets removed from `m` so that we can still have a solution.

In the end, we get a solution of the form `init`' mod `m` / c, such that `init` = `init`' · c + d mod `m` where 0 ≤ deg(d) < deg(c).

### Doing the rest

`xorout` is now easy to calculate modulo `m`:

&nbsp; a<sub>1</sub> - `init` · x<sup>`len(f₁)`</sup> = (`init` · x<sup>`len(f₁)`</sup> + `xorout`) - `init` · x<sup>`len(f₁)`</sup>  ≡ `xorout` mod `m`

`m` might still be bigger than `poly`, however it is not much.
Usually all other spurious factors are removed by this point (depending on how many files are given), but it is very improbable that the degree is bigger than a few 100 at most.
This can be very easily factored with modern factorization algorithms.

With the factors, it is a matter of simply picking those factors whose degrees add up to `width`, the subset sum problem.
Again, this is very fast at the scale of typical solutions.

For each possible solution `poly`, one can then take the values that are still modulo `m` and simply calculate them modulo `poly`.

For `refin` and `refout`, it is easiest to just run the whole algorithm for all the 4 value combinations.


Reverse Engineering the Checksum Range from Files and Their Corresponding Checksums with a Given Checksum Algorithm
-------------------------------------------------------------------------------------------------------------------
Suppose you have some files `f`<sub>k</sub> with k ranging from 1 to n.
Additionally, you have their corresponding checksums `crc`<sub>k</sub> using some known CRC algorithm.
But you do not know over what ranges of the file the checksums are taken.

All following operations are modulo `poly`.
Also, x = 10<sup>8</sup> (a shift equivalent to a byte, not a bit) in this context.

Let the pure checksum (without `init` or `xorout`) starting at byte a and ending before byte b of file `f`<sub>k</sub> be denoted by `[a..b]`<sub>k</sub>.
Then the actual checksum is

&nbsp; `crc`<sub>k</sub> = `[a..b]`<sub>k</sub> + `init` · x<sup>b - a</sup> + `xorout`

Then with

&nbsp; start(`m`) = (`[0..m]`<sub>k</sub> - `init`) · x<sup>`len(f_k)` - `m`</sup>\
&nbsp; end(`m`) = (`[0..m]`<sub>k</sub>  + `xorout` - `crc`<sub>k</sub>) · x<sup>`len(f_k)` - `m`</sup>

we can calculate for a the start of the sum and b the end of the sum

&nbsp; end(b) - start(a) = (`[0..b]`<sub>k</sub>  + `xorout` - `crc`<sub>k</sub>) · x<sup>`len(f_k)` - b</sup> - (`[0..a]`<sub>k</sub> - `init`) · x<sup>`len(f_k)` - a</sup>\
&nbsp; = (`[0..b]`<sub>k</sub>  + `xorout` - `crc`<sub>k</sub>) · x<sup>`len(f_k)` - b</sup> - (`[0..a]`<sub>k</sub> - `init`) · x<sup>b - a</sup>x<sup>`len(f_k)` - b</sup>\
&nbsp; = (`[0..b]`<sub>k</sub>  + `xorout` - `crc`<sub>k</sub> - (`[0..a]`<sub>k</sub> - `init`) · x<sup>b - a</sup>) · x<sup>`len(f_k)` - b</sup>\
&nbsp; = (`[a..b]`<sub>k</sub>  + `xorout` - `crc`<sub>k</sub> + `init` · x<sup>b - a</sup>) · x<sup>`len(f_k)` - b</sup>

Note that x<sup>n</sup> is never 0 since `poly` does not contain a factor of 10 anywhere.
Therefore, since `[a..b]`<sub>k</sub> + `init` · x<sup>b - a</sub> + `xorout` is the checksum from a to b, the above equation should be 0 iff it is equal to `crc`<sub>k</sub>.
That is equivalent to end(b) = start(a), so we just have to find values of start(a) that are equal to end(a).

All start(a) and end(a) values can be calculated in linear time:
if one has `[0..a]` one can of course calculate `[0..a+1]` in constant time, which means one can first calculate all (`[0..m]`<sub>k</sub> - `init`) and (`[0..m]`<sub>k</sub> + `xorout` - `crc`<sub>k</sub>) in linear time.
After that, one can then calculate the whole terms, this time in reverse directions, because one can calculate x<sup>a + 1</sup> from x<sup>a</sup> in constant time.

Using a hashtable to find duplicates, one can then get all pairs (a, b) where start(a) = end(b), again in linear time. One has to sometimes represent the offsets for example as `a₁,a₂,a₃:b₁,b₂`  meaning that the intervals may start at any of the a<sub>k</sub> and end at any of the b<sub>k</sub> to avoid quadratically many pairs when explicitely showing them.

This can be applied to multiple files at once by basically just doing everything in (𝔽[X]/`poly`)<sup>n</sup> instead of plain 𝔽[X]/`poly`.