.. _changelog:

=========
CHANGELOG
=========


.. _changelog-v0.0.6:

v0.0.6 (2025-04-16)
===================

Bug Fixes
---------

* fix: fix pypi yml (`e425d80`_)

* fix: update defaults for timing (`0e8eaee`_)

Unknown
-------

* fix:update build (`d087344`_)

* add: benchmarking tools (`d3badda`_)

* add: benchmarking tools (`f874d58`_)

* add ability to pass tuned parms (`f9a8a3d`_)

* np.Nan --> np.nan (`982d178`_)

* fix the plot for coherence for ET case

1. In the demo compute_psd, the factor applied to psd_all has been removed.
2. In the demo plot_coherence, the color of the uniform CI has been adjusted so that both the pointwise CI and uniform CI share the same color but with different shades. (`4b6583b`_)

* add R comparison scripts (`95d137c`_)

* refactor true_fmax / original_fmax rescaling for readability (`a48933e`_)

* remove mean notes (`b3b1586`_)

* Merge branch 'main' of github.com:nz-gravity/sgvb_psd (`0f148e6`_)

* Merge branch 'main' of https://github.com/nz-gravity/sgvb_psd (`742e43d`_)

* Update sim_varma.py (`6ed433f`_)

* Update sim_varma.py

Ensure that the periodogram and the frequency are of equal length. (`12dbeb2`_)

* reduce mean for each chunk

The mean for each chunk of the dataset is reduced.
After the Fourier transform for each chunk, the 0 frequency and the corresponding data are removed. (`b6bdb7c`_)

* Update analysis_data.py

x is a multi-dimensional array, so np.mean(x, axis=0) and np.std(x, axis=0) return arrays, hence requiring np.any(). (`6db45da`_)

* add notes on mean parameters (`76e30be`_)

* Update analysis_data.py (`15f70fb`_)

* add uniform CI for coherence

1. For plot_coherence, the code for uniform CI is added.
2. For plot_losses, the order of the subplots is changed, now the first subplot is the losses for the MAP, the second is for ELBO.
3.For test_end_to_end, just change some parameters to test the psd and coh, the simulation time is still within 50s. (`d83884a`_)

.. _e425d80: https://github.com/nz-gravity/sgvb_psd/commit/e425d8089cde0dff22e8d69919e5b43c0c110c20
.. _0e8eaee: https://github.com/nz-gravity/sgvb_psd/commit/0e8eaee3ad30bbef00aaca22d7fc85d360c52dca
.. _d087344: https://github.com/nz-gravity/sgvb_psd/commit/d087344c22566a8b5606722ef7284f0c63c8004e
.. _d3badda: https://github.com/nz-gravity/sgvb_psd/commit/d3badda922a438ae667a9207973d61ed21d9d93b
.. _f874d58: https://github.com/nz-gravity/sgvb_psd/commit/f874d586651aa4a83ab40191ea08faaa84697808
.. _f9a8a3d: https://github.com/nz-gravity/sgvb_psd/commit/f9a8a3dea52197da853758190fd7011707941b05
.. _982d178: https://github.com/nz-gravity/sgvb_psd/commit/982d178d1b04d1fc33ad28f74df3870786953808
.. _4b6583b: https://github.com/nz-gravity/sgvb_psd/commit/4b6583bc673938e36b23edce3c107834f9706c43
.. _95d137c: https://github.com/nz-gravity/sgvb_psd/commit/95d137c147ebfa1557789a7f259d838d04cead13
.. _a48933e: https://github.com/nz-gravity/sgvb_psd/commit/a48933e04495d06b7d5bce2d58a4e02a0fa7a968
.. _b3b1586: https://github.com/nz-gravity/sgvb_psd/commit/b3b15864899a38088692ed49299e4eb298c37d9c
.. _0f148e6: https://github.com/nz-gravity/sgvb_psd/commit/0f148e6dfbf67588fcae5c0aeebf734265449f07
.. _742e43d: https://github.com/nz-gravity/sgvb_psd/commit/742e43dc0a7b46bc203599d90e8a22a912378c95
.. _6ed433f: https://github.com/nz-gravity/sgvb_psd/commit/6ed433fe02f65dcafc9c6f143dbf9611aebc2692
.. _12dbeb2: https://github.com/nz-gravity/sgvb_psd/commit/12dbeb2d42e8cca2d25d41aea415fd4887414315
.. _b6bdb7c: https://github.com/nz-gravity/sgvb_psd/commit/b6bdb7c6778cbb4290148dd8092b0162cd8fea2c
.. _6db45da: https://github.com/nz-gravity/sgvb_psd/commit/6db45dab8d09a5461347c2f07cca7a5cb877b479
.. _76e30be: https://github.com/nz-gravity/sgvb_psd/commit/76e30be3b02ca1f9ca6f2cfc7a884e4d068535aa
.. _15f70fb: https://github.com/nz-gravity/sgvb_psd/commit/15f70fb0b33b862c12bf1728f3cb5e053b15834a
.. _d83884a: https://github.com/nz-gravity/sgvb_psd/commit/d83884a0629d4ea94ec2f59d76b3147e78fb45d9


.. _changelog-v0.0.5:

v0.0.5 (2024-09-30)
===================

Unknown
-------

* v0.0.5 (`505482e`_)

* ET study slurm (`cde1b58`_)

* fix the opposite uniform CI for imag part

Now the uniform CI for the imag cross spectrum have the same direction. (`a534297`_)

* add comment on fs/2 rescaling (`493fc1b`_)

* change psd_scaling and psd_offset from (axis=0) to (), ie scaling is a scalar now, not a p-dim vector (`9667e74`_)

* Merge pull request #5 from nz-gravity/refactoring

Refactoring to help with GPU + different CI (`0529cdc`_)

* allow ls as an option (`fba7a6f`_)

* add fixed sim study (`e62e175`_)

* add different quantiles (`907dc42`_)

* refactor CI (`e236a7d`_)

* add chunks (`b82ec18`_)

* Merge branch 'main' into refactoring (`cf37eac`_)

* add ET runtime (`3d551c3`_)

* Merge branch 'main' into refactoring (`147830f`_)

* add hard-runtimes for tests (`ede0bc1`_)

* start refactoring (`f4bdbfc`_)

* a typo, suppose to plot uniform CI (`75ebb8e`_)

* add code to create uniform CI (`a0af8ba`_)

* Update _toc.yml (`919a720`_)

* Update index.md (`86334c8`_)

* Update README.md with arxiv (`dfd3cc1`_)

* add Pypi badge (`7d07e4f`_)

* add Pypi badge (`a932243`_)

* add plotting fixes (`684e776`_)

* refactor (`20ccaee`_)

* add JAX-jit (`7aedc5a`_)

* Merge branch 'main' of github.com:nz-gravity/sgvb_psd into main (`3bbcea0`_)

.. _505482e: https://github.com/nz-gravity/sgvb_psd/commit/505482e0892b4f92c3350187a9ca1b3d4839efdb
.. _cde1b58: https://github.com/nz-gravity/sgvb_psd/commit/cde1b58109241485f1882dc09d2b8ed315c9b641
.. _a534297: https://github.com/nz-gravity/sgvb_psd/commit/a53429786ce55eda799f30a17df193904a42fcd6
.. _493fc1b: https://github.com/nz-gravity/sgvb_psd/commit/493fc1bacc79a235973fcf3b970849f9586018b1
.. _9667e74: https://github.com/nz-gravity/sgvb_psd/commit/9667e74f8efc08112a8a83743420fd172ec5fa80
.. _0529cdc: https://github.com/nz-gravity/sgvb_psd/commit/0529cdca6214c3dbd026148651d6caf87683b26b
.. _fba7a6f: https://github.com/nz-gravity/sgvb_psd/commit/fba7a6fbb703ee3fe9622ded012dee914c110375
.. _e62e175: https://github.com/nz-gravity/sgvb_psd/commit/e62e17538033cf440a69776d54e112261408c84a
.. _907dc42: https://github.com/nz-gravity/sgvb_psd/commit/907dc421463f4736556f9f18c4e37d0c8764a418
.. _e236a7d: https://github.com/nz-gravity/sgvb_psd/commit/e236a7dee11a2c50ccadbc1c0db2a1cfb588b1a6
.. _b82ec18: https://github.com/nz-gravity/sgvb_psd/commit/b82ec1822135efccf322d1cccc346b62959fcd52
.. _cf37eac: https://github.com/nz-gravity/sgvb_psd/commit/cf37eace0fc49c9f17d7fd4ac3bec5f2c7cb587a
.. _3d551c3: https://github.com/nz-gravity/sgvb_psd/commit/3d551c34fa8a53e156daa004fb41f9d7b3d81235
.. _147830f: https://github.com/nz-gravity/sgvb_psd/commit/147830fccbeeec9c8fec6b8c4f7a7d8fa0d27108
.. _ede0bc1: https://github.com/nz-gravity/sgvb_psd/commit/ede0bc1850d46a834c7bab786c27d24d235058db
.. _f4bdbfc: https://github.com/nz-gravity/sgvb_psd/commit/f4bdbfc38e072a72b4c543f5aa359716126f3ca6
.. _75ebb8e: https://github.com/nz-gravity/sgvb_psd/commit/75ebb8e24e55a94be0fdf94b704e971b26a2e591
.. _a0af8ba: https://github.com/nz-gravity/sgvb_psd/commit/a0af8ba4768de824d14680d92e3180e9cbb0219a
.. _919a720: https://github.com/nz-gravity/sgvb_psd/commit/919a720f2f00828498153b8f44d082b7b20cc83c
.. _86334c8: https://github.com/nz-gravity/sgvb_psd/commit/86334c87923e94ec1266b0c52bb5e64339b2cb66
.. _dfd3cc1: https://github.com/nz-gravity/sgvb_psd/commit/dfd3cc17479ce8aa797285c0e9dda7bc0a055190
.. _7d07e4f: https://github.com/nz-gravity/sgvb_psd/commit/7d07e4f4aaf55d2c6acae72988df08371f0ab7f5
.. _a932243: https://github.com/nz-gravity/sgvb_psd/commit/a93224301e0ecc109e70c2b3d08f62559fc066eb
.. _684e776: https://github.com/nz-gravity/sgvb_psd/commit/684e7764f43a3bea3040009c9726399a27c99d40
.. _20ccaee: https://github.com/nz-gravity/sgvb_psd/commit/20ccaee622bc7f9ef2e05e8d2a5b948c4b393a5b
.. _7aedc5a: https://github.com/nz-gravity/sgvb_psd/commit/7aedc5a98669bac974e03a8067f6e838fee081e9
.. _3bbcea0: https://github.com/nz-gravity/sgvb_psd/commit/3bbcea08222595c5b6e73264bce87e39fd9dcea7


.. _changelog-v0.0.4:

v0.0.4 (2024-09-23)
===================

Unknown
-------

* add v0.0.4 (`fac5f8f`_)

* Merge branch 'main' of github.com:nz-gravity/sgvb_psd (`6af7c36`_)

* Update pyproject.toml (`7a55afe`_)

* add simulation study example (`f658fa4`_)

* add example scripts (`7cffb71`_)

* v0.0.4 (`2ba266f`_)

.. _fac5f8f: https://github.com/nz-gravity/sgvb_psd/commit/fac5f8facf5cd2e0dc08135505b28401a02de64c
.. _6af7c36: https://github.com/nz-gravity/sgvb_psd/commit/6af7c362b80b225cede5a598fe08a0bd771be02e
.. _7a55afe: https://github.com/nz-gravity/sgvb_psd/commit/7a55afe78df273a5e3b47cc20160516dd9a6cdfd
.. _f658fa4: https://github.com/nz-gravity/sgvb_psd/commit/f658fa4eb1e33084f17e3c244dbeefec7c58d004
.. _7cffb71: https://github.com/nz-gravity/sgvb_psd/commit/7cffb71f891d126d9ae776d93074d97dfa54f3df
.. _2ba266f: https://github.com/nz-gravity/sgvb_psd/commit/2ba266f96fe99bc817da7e8de4fab6897ce25849


.. _changelog-v0.0.3:

v0.0.3 (2024-09-21)
===================

Unknown
-------

* v0.0.3 (`1e34b46`_)

* fix matplolib rc file loc (`4f99698`_)

* add dta to pyproj (`06ceefc`_)

.. _1e34b46: https://github.com/nz-gravity/sgvb_psd/commit/1e34b4655f0168baf061a73e790aa40c6e6f2587
.. _4f99698: https://github.com/nz-gravity/sgvb_psd/commit/4f99698699a7e7ba823aae81189716206fe0eeac
.. _06ceefc: https://github.com/nz-gravity/sgvb_psd/commit/06ceefc4450b2e6d41b46552640ec7ccec5c0556


.. _changelog-v0.0.2:

v0.0.2 (2024-09-20)
===================

Unknown
-------

* V0.0.2 release (`f3bf4c1`_)

.. _f3bf4c1: https://github.com/nz-gravity/sgvb_psd/commit/f3bf4c1a446399210f5336c9bbf0579d8b537729


.. _changelog-v0.0.1:

v0.0.1 (2024-09-20)
===================

Unknown
-------

* add details to readme (`9d8eb0c`_)

* fix plots (`25f32b8`_)

* Adjust plots (`02da83d`_)

* Merge branch 'main' of github.com:nz-gravity/sgvb_psd (`d9aaf83`_)

* Update docs_and_tests.yml (`5a46f49`_)

* Update docs_and_tests.yml (`5b7810c`_)

* add plots (`e3cd54b`_)

* fix notebooks (`76e433f`_)

* add nm steps for opt (`62a1ec3`_)

* add main study (`d24aa5b`_)

* fix worklow (`7afb76e`_)

* Merge branch 'main' of github.com:nz-gravity/sgvb_psd into main (`da93afc`_)

* add documentation (`59b39b6`_)

* refrmat (`4346537`_)

* add logo (`fbaf4ec`_)

* Update README.md (`ab27e5b`_)

* Create CITATION.cff (`1f06d53`_)

* rerun sims (`e183aa6`_)

* remove junk (`48e5a1d`_)

* Merge branch 'main' of github.com:nz-gravity/sgvb_psd (`1b26171`_)

* add smaller dataset (`d2c1aed`_)

* clean examples (`5de14b7`_)

* refactoring (`5406d07`_)

* add example (`e270a89`_)

* add test (`47ce5ae`_)

* add fixes (`08e8a93`_)

* fix the issue of storing all samples

fix the issue of storing all samples during optimization (`41baf92`_)

* add pythhon scirpt (`8584796`_)

* added best LR log (`599d02c`_)

* remove duration from specVI (`188d4ea`_)

* add fixed psd generator (`02bcede`_)

* fix plot scaling (`a5e3f7b`_)

* some changes to the freq-ranges (`e14d054`_)

* acking on max f (`732ba02`_)

* chunk simulation test (`0074754`_)

* add coh plot (`b5d29aa`_)

* set the x limit for test_ET (`82cbdbc`_)

* add package (`e73ac97`_)

* add notes on hyperopt (`2233fcf`_)

* fixing logs (`68cd59b`_)

* add the code for the squared coherence (`ae54f60`_)

* fix the plot for ET psd (`55a3f34`_)

* add ET test (`6799780`_)

* PSD Analyzer is modified

Under the test_simulation, it is able to find the L2 errors, coverages etc between  the true psd and the estimated psd. (`55f0244`_)

* fix the scaling for var(2)

now the plot for the true psd is fixed, matches with the estimated psd (`6b520e8`_)

* change latex to html (`653c1ad`_)

* fix CI (`d61ca9a`_)

* fix html rendering (`80c3bed`_)

* add plots to docs (`39c9f79`_)

* fix formatting (`de07a69`_)

* add simulation study plot (`503500f`_)

* hacking with jianan (`b408f13`_)

* period, dataset and scale fixed (`4dff16b`_)

* add todos for jianan (`69fc873`_)

* fix plotting (`7e9be8f`_)

* add notes (`45717f3`_)

* setup (`972858b`_)

* pick half of the frequency domain (`312408d`_)

* hacking on psd plotting and samplig freq and periodogram with jianan (`fbe2f61`_)

* this should be defined under the init (`161fdf8`_)

* self.sampling_freq is defined

Now the freq works for both simulation study (freq form 0 to pi) and ET case (freq from 0 to 128Hz). (`94307fe`_)

* hacking with jianan on varma (`d955539`_)

* fix formatting (`c2cf2fc`_)

* create psd analyzer class

This class can used by both var2 and vma1 models for compute the L2 errors, CI length and coverages. Reduce the repetitions for the same code. (`977a36a`_)

* erase the option under sparse_op

the blocked log likelihood is only used under the default condition i.e., sparse_op = false, so no need to keep the code for when sparse_op = true. (`61cb9b3`_)

* create the class for psd construction

lr_tuner contains a class the construct the psd under the optimised lr by Hyperopt. var2_256_errors_....can import it and get the est psd for comparisons. (`58ce132`_)

* setup specVi and examples (`9480ad0`_)

* add failing tests (`013efbb`_)

* setup docs (`869134b`_)

* init repo (`89cfb7c`_)

* Initial commit (`3fd85f1`_)

* remove large files (`d8b2f51`_)

* add smaller dataset (`fcb1143`_)

* clean examples (`db3e2dc`_)

* refactoring (`defc92b`_)

* add example (`605c0ef`_)

* add test (`180a65e`_)

* add fixes (`92f3efe`_)

* fix the issue of storing all samples

fix the issue of storing all samples during optimization (`e6a159b`_)

* add pythhon scirpt (`89567eb`_)

* added best LR log (`bb207e2`_)

* remove duration from specVI (`8434cca`_)

* add fixed psd generator (`5872f48`_)

* fix plot scaling (`73d2b33`_)

* some changes to the freq-ranges (`09cdd0c`_)

* acking on max f (`3bc8469`_)

* chunk simulation test (`c62dd11`_)

* add coh plot (`f916b13`_)

* set the x limit for test_ET (`3a82440`_)

* add package (`58739a5`_)

* add notes on hyperopt (`55c264f`_)

* fixing logs (`bf1ffd4`_)

* add the code for the squared coherence (`4f555f4`_)

* fix the plot for ET psd (`8a86fa2`_)

* add ET test (`8181cda`_)

* PSD Analyzer is modified

Under the test_simulation, it is able to find the L2 errors, coverages etc between  the true psd and the estimated psd. (`3608e27`_)

* fix the scaling for var(2)

now the plot for the true psd is fixed, matches with the estimated psd (`8ebf9f1`_)

* change latex to html (`c7798dd`_)

* fix CI (`a74f9fa`_)

* fix html rendering (`7648607`_)

* add plots to docs (`11285b5`_)

* fix formatting (`4aaf9aa`_)

* add simulation study plot (`7c83a83`_)

* hacking with jianan (`ae5bd4d`_)

* period, dataset and scale fixed (`a827431`_)

* add todos for jianan (`aca2f94`_)

* fix plotting (`bde3197`_)

* add notes (`d5e9376`_)

* setup (`01b17ec`_)

* pick half of the frequency domain (`fb4b3aa`_)

* hacking on psd plotting and samplig freq and periodogram with jianan (`79c29b1`_)

* this should be defined under the init (`c497f61`_)

* self.sampling_freq is defined

Now the freq works for both simulation study (freq form 0 to pi) and ET case (freq from 0 to 128Hz). (`af05daf`_)

* hacking with jianan on varma (`0fa7bbd`_)

* fix formatting (`29f64eb`_)

* create psd analyzer class

This class can used by both var2 and vma1 models for compute the L2 errors, CI length and coverages. Reduce the repetitions for the same code. (`bb95cef`_)

* erase the option under sparse_op

the blocked log likelihood is only used under the default condition i.e., sparse_op = false, so no need to keep the code for when sparse_op = true. (`2525537`_)

* create the class for psd construction

lr_tuner contains a class the construct the psd under the optimised lr by Hyperopt. var2_256_errors_....can import it and get the est psd for comparisons. (`0a3869f`_)

* setup specVi and examples (`1f5aa7c`_)

* add failing tests (`1ba7604`_)

* setup docs (`f82493a`_)

* init repo (`ec740fe`_)

* Initial commit (`e2a5ebf`_)

.. _9d8eb0c: https://github.com/nz-gravity/sgvb_psd/commit/9d8eb0c7483669e1c22ceff384a3407b2c1e621e
.. _25f32b8: https://github.com/nz-gravity/sgvb_psd/commit/25f32b855b8f79ef753016439a1a3c28095aad66
.. _02da83d: https://github.com/nz-gravity/sgvb_psd/commit/02da83d69dd169715162735e24aff386901dbae0
.. _d9aaf83: https://github.com/nz-gravity/sgvb_psd/commit/d9aaf83f61a30c95ffd507b8fc7033cf0cad7950
.. _5a46f49: https://github.com/nz-gravity/sgvb_psd/commit/5a46f49355d23b15cb808dfc8fa17bd23f8da615
.. _5b7810c: https://github.com/nz-gravity/sgvb_psd/commit/5b7810c40faead073df9210c1158fafcc4cb923c
.. _e3cd54b: https://github.com/nz-gravity/sgvb_psd/commit/e3cd54bf2561d98ed065b456ecb33bf9c023a953
.. _76e433f: https://github.com/nz-gravity/sgvb_psd/commit/76e433f5334984d10fff101da1e734b399841b2d
.. _62a1ec3: https://github.com/nz-gravity/sgvb_psd/commit/62a1ec3eb81d03cc25d25091fc1634990c1cc709
.. _d24aa5b: https://github.com/nz-gravity/sgvb_psd/commit/d24aa5b34ae631182aa12625fc88ed781e78f4a7
.. _7afb76e: https://github.com/nz-gravity/sgvb_psd/commit/7afb76e401ff897cc81efff0492423c2b55a926c
.. _da93afc: https://github.com/nz-gravity/sgvb_psd/commit/da93afc32cdd7a4f72ff93757c1c97ca1c14352e
.. _59b39b6: https://github.com/nz-gravity/sgvb_psd/commit/59b39b6fa8e8cec312124d3a7109e5996ed3f228
.. _4346537: https://github.com/nz-gravity/sgvb_psd/commit/434653710602d03d1499eb1ce7c2b63b95a9affc
.. _fbaf4ec: https://github.com/nz-gravity/sgvb_psd/commit/fbaf4ecf8524a862bc504d9d455d46edcec119e1
.. _ab27e5b: https://github.com/nz-gravity/sgvb_psd/commit/ab27e5b42efe58c342a760e232bb758d20aa99ad
.. _1f06d53: https://github.com/nz-gravity/sgvb_psd/commit/1f06d533f8446cd0d1e1dffc3a00666848cfc44c
.. _e183aa6: https://github.com/nz-gravity/sgvb_psd/commit/e183aa60ea4a9ba41b2ec24edfa38d31b5b82e4a
.. _48e5a1d: https://github.com/nz-gravity/sgvb_psd/commit/48e5a1d4f32996509f9dc39155b35d0574618d73
.. _1b26171: https://github.com/nz-gravity/sgvb_psd/commit/1b26171d7c7616d15e1a7266a6a3828a005e9d06
.. _d2c1aed: https://github.com/nz-gravity/sgvb_psd/commit/d2c1aedfff0db5a3bf4df1b9fb343e35345c32aa
.. _5de14b7: https://github.com/nz-gravity/sgvb_psd/commit/5de14b783cbec3dc5541571e0a3e9f500b4c9416
.. _5406d07: https://github.com/nz-gravity/sgvb_psd/commit/5406d070d684da3479be1cdf9e81b5c61741759f
.. _e270a89: https://github.com/nz-gravity/sgvb_psd/commit/e270a8947c48163983ac5f88104231c56f84555c
.. _47ce5ae: https://github.com/nz-gravity/sgvb_psd/commit/47ce5ae575b97df7344174a92a9f56b849644545
.. _08e8a93: https://github.com/nz-gravity/sgvb_psd/commit/08e8a93020ee82c39ea260587e9ee2a0fbea97ac
.. _41baf92: https://github.com/nz-gravity/sgvb_psd/commit/41baf922526548fbdfe184e8147810deac84beba
.. _8584796: https://github.com/nz-gravity/sgvb_psd/commit/85847967230a78297db8fbcebbdc5ccfd427340d
.. _599d02c: https://github.com/nz-gravity/sgvb_psd/commit/599d02c459c94ccbed6f78522b091a222e572692
.. _188d4ea: https://github.com/nz-gravity/sgvb_psd/commit/188d4ea76be58ddd3507d26a683a88642f4d5dbe
.. _02bcede: https://github.com/nz-gravity/sgvb_psd/commit/02bcedeeeb92225396cc4f2eb447a083ecf6dcd6
.. _a5e3f7b: https://github.com/nz-gravity/sgvb_psd/commit/a5e3f7b2ead79777db438c1b270491a262a72aa4
.. _e14d054: https://github.com/nz-gravity/sgvb_psd/commit/e14d054f07c227ba5e8b698c74b432a8bc4661ff
.. _732ba02: https://github.com/nz-gravity/sgvb_psd/commit/732ba02da096aba5c404b5be2f3597cc064d6155
.. _0074754: https://github.com/nz-gravity/sgvb_psd/commit/0074754726e15ef8574430528e142831a230f758
.. _b5d29aa: https://github.com/nz-gravity/sgvb_psd/commit/b5d29aa524da3f1a26dee9d3f7a748c179b23920
.. _82cbdbc: https://github.com/nz-gravity/sgvb_psd/commit/82cbdbc069cd622eca1d1a1ae9742d7492111b2b
.. _e73ac97: https://github.com/nz-gravity/sgvb_psd/commit/e73ac979f5501856f24e3aac406ca54073002bda
.. _2233fcf: https://github.com/nz-gravity/sgvb_psd/commit/2233fcfb5e3a8d82f26bdf1affb6342465d2c776
.. _68cd59b: https://github.com/nz-gravity/sgvb_psd/commit/68cd59bc520053f1cba7de409595961f57e5de13
.. _ae54f60: https://github.com/nz-gravity/sgvb_psd/commit/ae54f60be8224bf90fa88ec2c0467716607bfacb
.. _55a3f34: https://github.com/nz-gravity/sgvb_psd/commit/55a3f34527cb1eb3db1072fe546a22ff3a796e5e
.. _6799780: https://github.com/nz-gravity/sgvb_psd/commit/6799780b60aff60005ed1362b3fe1e1cba199738
.. _55f0244: https://github.com/nz-gravity/sgvb_psd/commit/55f0244a221b5e1be2d9daacbe9a5dc6f8d71a9f
.. _6b520e8: https://github.com/nz-gravity/sgvb_psd/commit/6b520e8e273e12e5d2dd9470e19d7c082fd80212
.. _653c1ad: https://github.com/nz-gravity/sgvb_psd/commit/653c1ad6292caf8634f3f89c6677c29de0bfa63c
.. _d61ca9a: https://github.com/nz-gravity/sgvb_psd/commit/d61ca9a2b87b5c56026d917098ff895459d01bd6
.. _80c3bed: https://github.com/nz-gravity/sgvb_psd/commit/80c3bedfc38e4338bc32546288d5f11f71e540f4
.. _39c9f79: https://github.com/nz-gravity/sgvb_psd/commit/39c9f798b6fbd028de6172e7bc135a375054e2e4
.. _de07a69: https://github.com/nz-gravity/sgvb_psd/commit/de07a69f2eb7d74c5ce14ed63402caf5d86eadc7
.. _503500f: https://github.com/nz-gravity/sgvb_psd/commit/503500fccc23e0d43b96b4182e6e2335d658478b
.. _b408f13: https://github.com/nz-gravity/sgvb_psd/commit/b408f138cdd4e438edeba377817c6abbbdebd617
.. _4dff16b: https://github.com/nz-gravity/sgvb_psd/commit/4dff16b01e66f386a83a4fba7c5b5d7b9a7e1c51
.. _69fc873: https://github.com/nz-gravity/sgvb_psd/commit/69fc873c5e5f17adae42e8cb5f6850433c49c782
.. _7e9be8f: https://github.com/nz-gravity/sgvb_psd/commit/7e9be8fa27e4aa1ddb1d5a3f2fa5055d5f6300b4
.. _45717f3: https://github.com/nz-gravity/sgvb_psd/commit/45717f30c338f8b05a7c8afb93ffde61ea5e849a
.. _972858b: https://github.com/nz-gravity/sgvb_psd/commit/972858b0cad4e75fa583c17fee69c0ed02ff0c30
.. _312408d: https://github.com/nz-gravity/sgvb_psd/commit/312408de4fbead05114488f612d09dcf5b3d336a
.. _fbe2f61: https://github.com/nz-gravity/sgvb_psd/commit/fbe2f618e4ff2e165853ec6cb33dd9640ce4030c
.. _161fdf8: https://github.com/nz-gravity/sgvb_psd/commit/161fdf83b801a5e41217097e47ae0f1763ce20aa
.. _94307fe: https://github.com/nz-gravity/sgvb_psd/commit/94307feeb9af64fe0d9ad7a2ed9a62e7c56493a4
.. _d955539: https://github.com/nz-gravity/sgvb_psd/commit/d9555393fb47814a71ea77d7906766c51a6a2c41
.. _c2cf2fc: https://github.com/nz-gravity/sgvb_psd/commit/c2cf2fc1ad7a982e9b44484b40672559d4c76f0a
.. _977a36a: https://github.com/nz-gravity/sgvb_psd/commit/977a36a79ab49620163bb573598a48119cdc8b0b
.. _61cb9b3: https://github.com/nz-gravity/sgvb_psd/commit/61cb9b39e78b365bea2cfcbf1325dbae6e898253
.. _58ce132: https://github.com/nz-gravity/sgvb_psd/commit/58ce13235243f4c90a3a994367614824b31f4a0a
.. _9480ad0: https://github.com/nz-gravity/sgvb_psd/commit/9480ad0762c035612a8ceb0f1d0f53ca1962568d
.. _013efbb: https://github.com/nz-gravity/sgvb_psd/commit/013efbb8f8605ddee111c5bc3bb95cc94eda6747
.. _869134b: https://github.com/nz-gravity/sgvb_psd/commit/869134beb28ee5cfc34f88b31836953425e468df
.. _89cfb7c: https://github.com/nz-gravity/sgvb_psd/commit/89cfb7c56f7ff1c1a2d99c460d2d0ed5ee356c83
.. _3fd85f1: https://github.com/nz-gravity/sgvb_psd/commit/3fd85f1d3f1d2097db083bb589e75e1fcb380d87
.. _d8b2f51: https://github.com/nz-gravity/sgvb_psd/commit/d8b2f51aba58cf9cfef5be55b6abe0516fe0308b
.. _fcb1143: https://github.com/nz-gravity/sgvb_psd/commit/fcb114310d59b02c531a8a9d883e3ced66994dd0
.. _db3e2dc: https://github.com/nz-gravity/sgvb_psd/commit/db3e2dcaac0ec14837b57e1181a53436cfb49ef4
.. _defc92b: https://github.com/nz-gravity/sgvb_psd/commit/defc92bc910fbab72da6b49d7e3d1fb5a7b3f0c7
.. _605c0ef: https://github.com/nz-gravity/sgvb_psd/commit/605c0ef4ba72415cd2b8db6f30c4448ae1237b38
.. _180a65e: https://github.com/nz-gravity/sgvb_psd/commit/180a65edbdfc69819e0c781f7e9aa3c7f5e93d52
.. _92f3efe: https://github.com/nz-gravity/sgvb_psd/commit/92f3efe89eddb0c5b41320e6ef9d924a1a4095ed
.. _e6a159b: https://github.com/nz-gravity/sgvb_psd/commit/e6a159b5a6aaf9682db71ad26f83b1e6c324a870
.. _89567eb: https://github.com/nz-gravity/sgvb_psd/commit/89567eb0c2ee7f7b7e80ff6b49c0730891a387e7
.. _bb207e2: https://github.com/nz-gravity/sgvb_psd/commit/bb207e245b1ad84d20f9dffc26c73f0c8e6905be
.. _8434cca: https://github.com/nz-gravity/sgvb_psd/commit/8434cca0cd757bec5e3e0bcb267517308c158a6d
.. _5872f48: https://github.com/nz-gravity/sgvb_psd/commit/5872f48fe0771a30b3fd0875129acf25adb405e5
.. _73d2b33: https://github.com/nz-gravity/sgvb_psd/commit/73d2b335eb6ab94d2af825a852ac1acd0ebbb5c3
.. _09cdd0c: https://github.com/nz-gravity/sgvb_psd/commit/09cdd0cd175426630ead9500e184e36156852888
.. _3bc8469: https://github.com/nz-gravity/sgvb_psd/commit/3bc846905d747f3b8a29aed6a2a4202110ac628e
.. _c62dd11: https://github.com/nz-gravity/sgvb_psd/commit/c62dd112c91a94fdc6886a5d82ccb0172ecf5aea
.. _f916b13: https://github.com/nz-gravity/sgvb_psd/commit/f916b1384974101825938f02b8d483e7c002edc2
.. _3a82440: https://github.com/nz-gravity/sgvb_psd/commit/3a8244021098a498bf285a6d71077e65117db730
.. _58739a5: https://github.com/nz-gravity/sgvb_psd/commit/58739a5626949a8ef0ccd40df333a167328d3d8b
.. _55c264f: https://github.com/nz-gravity/sgvb_psd/commit/55c264f137d0209800f814f9c2d6ec1c63e57226
.. _bf1ffd4: https://github.com/nz-gravity/sgvb_psd/commit/bf1ffd451d594466d9aea4845c1e5aa0f3594943
.. _4f555f4: https://github.com/nz-gravity/sgvb_psd/commit/4f555f4325b6f72aa3c9ecdad837cb4c076d234b
.. _8a86fa2: https://github.com/nz-gravity/sgvb_psd/commit/8a86fa2e0fb75b6aa2577826e5a924621d15fc1c
.. _8181cda: https://github.com/nz-gravity/sgvb_psd/commit/8181cda6370042b8a4203c9dcd71af089c41b78f
.. _3608e27: https://github.com/nz-gravity/sgvb_psd/commit/3608e27cc28fc0e0396325a54685e5b0d45b81d9
.. _8ebf9f1: https://github.com/nz-gravity/sgvb_psd/commit/8ebf9f163b7ca58f463de2bfb1cf889e04413aee
.. _c7798dd: https://github.com/nz-gravity/sgvb_psd/commit/c7798dd690a0d6ba4f61dacc6ed94234d5b2827c
.. _a74f9fa: https://github.com/nz-gravity/sgvb_psd/commit/a74f9fa0ce7495ee69c82c38f71f7ca6dfc9e10c
.. _7648607: https://github.com/nz-gravity/sgvb_psd/commit/764860782ae01c152537c1ed2aad8e5c3f9dcfa6
.. _11285b5: https://github.com/nz-gravity/sgvb_psd/commit/11285b54f2e660baff101c9da9bb017d7839e90f
.. _4aaf9aa: https://github.com/nz-gravity/sgvb_psd/commit/4aaf9aa86089045d5a63b163379e919b145d1213
.. _7c83a83: https://github.com/nz-gravity/sgvb_psd/commit/7c83a838420e54289053c30b71793226ccfed8f7
.. _ae5bd4d: https://github.com/nz-gravity/sgvb_psd/commit/ae5bd4d0967c5c6d4c38ccb86331ddad329cbaa9
.. _a827431: https://github.com/nz-gravity/sgvb_psd/commit/a827431f313e888b8171aa63adc7141db1b67594
.. _aca2f94: https://github.com/nz-gravity/sgvb_psd/commit/aca2f9410d07795b5556ca1bb7889d5062ebd038
.. _bde3197: https://github.com/nz-gravity/sgvb_psd/commit/bde3197f2c6cf24028db8236ac463b16e9abea80
.. _d5e9376: https://github.com/nz-gravity/sgvb_psd/commit/d5e937678db1fab6722db1ddb903f3215b13f243
.. _01b17ec: https://github.com/nz-gravity/sgvb_psd/commit/01b17ecf56b1116466de0327694794cfde403018
.. _fb4b3aa: https://github.com/nz-gravity/sgvb_psd/commit/fb4b3aa23ef94ebd2be95ba92676ed6d8c429cd2
.. _79c29b1: https://github.com/nz-gravity/sgvb_psd/commit/79c29b1b3b8da462de699a1f416de3f460a0a6ea
.. _c497f61: https://github.com/nz-gravity/sgvb_psd/commit/c497f61ac06a959115ab2471372876c3833dd369
.. _af05daf: https://github.com/nz-gravity/sgvb_psd/commit/af05daf7672b5c1a5d3e1cd576cecce452f46e16
.. _0fa7bbd: https://github.com/nz-gravity/sgvb_psd/commit/0fa7bbd49fd1185e6dcae0d513d087177affe971
.. _29f64eb: https://github.com/nz-gravity/sgvb_psd/commit/29f64eb2f78b033b7a07b95bbc514df36bbd6011
.. _bb95cef: https://github.com/nz-gravity/sgvb_psd/commit/bb95cefa3530aab3768cc3bf02b95d950aa074ed
.. _2525537: https://github.com/nz-gravity/sgvb_psd/commit/2525537748309db2bbdcdbcf42f75ecbb4bf3a61
.. _0a3869f: https://github.com/nz-gravity/sgvb_psd/commit/0a3869f31872e227e31e84e51181c1966afc4afc
.. _1f5aa7c: https://github.com/nz-gravity/sgvb_psd/commit/1f5aa7c878fc564d58ee5af9d20fa2175881a0fd
.. _1ba7604: https://github.com/nz-gravity/sgvb_psd/commit/1ba76046f16f4cf74142ea15aac6c27fd936c617
.. _f82493a: https://github.com/nz-gravity/sgvb_psd/commit/f82493aab55bcbc927cfb1609f6100112a0f4e19
.. _ec740fe: https://github.com/nz-gravity/sgvb_psd/commit/ec740fe3cee686b006b13c47b9eb6d4793d4a124
.. _e2a5ebf: https://github.com/nz-gravity/sgvb_psd/commit/e2a5ebf0e499caf952130acbeb4c12e4137f62b2
