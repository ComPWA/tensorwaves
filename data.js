window.BENCHMARK_DATA = {
  "lastUpdate": 1705082565751,
  "repoUrl": "https://github.com/ComPWA/tensorwaves",
  "entries": {
    "TensorWaves benchmark results": [
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "11079e5dca7d8a2fd1e6e4a2866b2832fe43bdc4",
          "message": "ci!: implement benchmark monitoring (#368)\n\n* ci: improve slow marker use\r\n* test!: add benchmark support with pytest-benchmark\r\n* ci: add workflow for benchmarks\r\n* ci: comment on commit if performance decreased\r\n* test: merge integration data test into benchmark\r\n* test: remove remaining integration test\r\n* test: move benchmarks to separate folder and unit tests to top\r\n* style: remove text from __init__.py files for mypy in test folders\r\n* test: add benchmarks for simple expression\r\n* fix: remove pytest-profiling\r\n* test: write unit test for CallbackList\r\n* docs: add link to continuous benchmarks\r\n* ci: update pip constraints and pre-commit config\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2021-12-03T21:42:08Z",
          "tree_id": "88096c7a7e5b6b26e9e703a6e8dbd9117ebe2108",
          "url": "https://github.com/ComPWA/tensorwaves/commit/11079e5dca7d8a2fd1e6e4a2866b2832fe43bdc4"
        },
        "date": 1638567971233,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22006834496217612,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.544042898000043 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4016680790106275,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.489617801999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.05921186102615,
            "unit": "iter/sec",
            "range": "stddev: 0.004405988686293936",
            "extra": "mean: 71.12774242858677 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 121.74917615797084,
            "unit": "iter/sec",
            "range": "stddev: 0.0008339967945119293",
            "extra": "mean: 8.213607940167822 msec\nrounds: 117"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.9175119817255744,
            "unit": "iter/sec",
            "range": "stddev: 0.12030755123435782",
            "extra": "mean: 342.7578040000185 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 74.02973942576587,
            "unit": "iter/sec",
            "range": "stddev: 0.0005783485626018681",
            "extra": "mean: 13.508084828567592 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.908434239124841,
            "unit": "iter/sec",
            "range": "stddev: 0.002022777491238156",
            "extra": "mean: 203.7309559999926 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 5.0219499429959225,
            "unit": "iter/sec",
            "range": "stddev: 0.0058801959834528525",
            "extra": "mean: 199.12583983332866 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.086688317251356,
            "unit": "iter/sec",
            "range": "stddev: 0.0037462513059051553",
            "extra": "mean: 196.59156166666017 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9493006797978816,
            "unit": "iter/sec",
            "range": "stddev: 0.01893599105750658",
            "extra": "mean: 1.0534070197999994 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.9358425074369645,
            "unit": "iter/sec",
            "range": "stddev: 0.004982303032589777",
            "extra": "mean: 202.59965720001674 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.07718305759696,
            "unit": "iter/sec",
            "range": "stddev: 0.003791646718113821",
            "extra": "mean: 196.95961099998271 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.157948353624968,
            "unit": "iter/sec",
            "range": "stddev: 0.003577744022928743",
            "extra": "mean: 193.87553566665852 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.0879407782135637,
            "unit": "iter/sec",
            "range": "stddev: 0.01768907071370034",
            "extra": "mean: 919.167678999986 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b1376237ac695bb9cef8133d4d0d8fbb853d946d",
          "message": "build: reduce dependencies in style requirements (#369)",
          "timestamp": "2021-12-03T21:52:13Z",
          "tree_id": "0b311b7ad5ff9565a2a8fef658d8c24e893fd68b",
          "url": "https://github.com/ComPWA/tensorwaves/commit/b1376237ac695bb9cef8133d4d0d8fbb853d946d"
        },
        "date": 1638568522427,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3024768026539775,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.3060386490000155 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4377934164259996,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.284182362000024 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.999010137880056,
            "unit": "iter/sec",
            "range": "stddev: 0.000516709263358909",
            "extra": "mean: 50.0024747777843 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 167.14253245746823,
            "unit": "iter/sec",
            "range": "stddev: 0.00007889204819314859",
            "extra": "mean: 5.982917605095304 msec\nrounds: 157"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.850715136716807,
            "unit": "iter/sec",
            "range": "stddev: 0.08799294828503812",
            "extra": "mean: 259.69202200000154 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 106.32800572449311,
            "unit": "iter/sec",
            "range": "stddev: 0.00014460132973470954",
            "extra": "mean: 9.40485992553179 msec\nrounds: 94"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.707041041097085,
            "unit": "iter/sec",
            "range": "stddev: 0.0010085005259978413",
            "extra": "mean: 129.7514824000018 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.037063241133366,
            "unit": "iter/sec",
            "range": "stddev: 0.0002558111333700164",
            "extra": "mean: 99.6307361999925 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.892778291633705,
            "unit": "iter/sec",
            "range": "stddev: 0.0008535359842260114",
            "extra": "mean: 101.08383818180755 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.6972692164753687,
            "unit": "iter/sec",
            "range": "stddev: 0.00092445843394973",
            "extra": "mean: 589.1817221999986 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.6690701619462205,
            "unit": "iter/sec",
            "range": "stddev: 0.0004991256016013146",
            "extra": "mean: 130.39390420001382 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.103218479489323,
            "unit": "iter/sec",
            "range": "stddev: 0.0018728882370524819",
            "extra": "mean: 109.85125779998839 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.099108405026882,
            "unit": "iter/sec",
            "range": "stddev: 0.000830812281120767",
            "extra": "mean: 109.90087770001082 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.8906038960159062,
            "unit": "iter/sec",
            "range": "stddev: 0.0033561658419269877",
            "extra": "mean: 528.9315239999837 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e1f00348ec7e64d60e9b12ed42d5b9e8dbf2aa97",
          "message": "ci: speed up pytest collect (#370)\n\n* ci: merge tox cov job into py job\r\n* ci: simplify tox pydeps job\r\n* fix: include doctests in coverage computation\r\n* fix: remove histogram from tox benchmark job\r\n  (causes problems with pygal and pygaljs)\r\n* test: speed up pytest collect with inline imports",
          "timestamp": "2021-12-05T19:45:18Z",
          "tree_id": "4e282fff6e904072d4de7eebee22cd14658329ab",
          "url": "https://github.com/ComPWA/tensorwaves/commit/e1f00348ec7e64d60e9b12ed42d5b9e8dbf2aa97"
        },
        "date": 1638733724725,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.27127499866795823,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.686296212000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.47157658947752057,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.1205463170000485 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.555062413280645,
            "unit": "iter/sec",
            "range": "stddev: 0.000619721139062683",
            "extra": "mean: 53.89364787500028 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 169.14005209381563,
            "unit": "iter/sec",
            "range": "stddev: 0.00011521332553971961",
            "extra": "mean: 5.912260210522684 msec\nrounds: 152"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.517762419653576,
            "unit": "iter/sec",
            "range": "stddev: 0.12166451310712155",
            "extra": "mean: 284.27161380002417 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 103.49136705025907,
            "unit": "iter/sec",
            "range": "stddev: 0.00016061194032063184",
            "extra": "mean: 9.662641711113592 msec\nrounds: 90"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.478657575283733,
            "unit": "iter/sec",
            "range": "stddev: 0.0009250036718596304",
            "extra": "mean: 133.7138369999593 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.419761467204406,
            "unit": "iter/sec",
            "range": "stddev: 0.0002822554360154693",
            "extra": "mean: 106.15980070000433 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.195962064457998,
            "unit": "iter/sec",
            "range": "stddev: 0.001392515413492101",
            "extra": "mean: 108.74338030002946 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.5420480599355326,
            "unit": "iter/sec",
            "range": "stddev: 0.0010740955709193718",
            "extra": "mean: 648.4882190000008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.910334763317306,
            "unit": "iter/sec",
            "range": "stddev: 0.0002583141335208369",
            "extra": "mean: 144.7107896000034 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.729956686320199,
            "unit": "iter/sec",
            "range": "stddev: 0.00011060590638758692",
            "extra": "mean: 114.54810555555166 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.642556174959305,
            "unit": "iter/sec",
            "range": "stddev: 0.0013474928869228625",
            "extra": "mean: 115.7065085555789 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.7430575416049714,
            "unit": "iter/sec",
            "range": "stddev: 0.002141470746760797",
            "extra": "mean: 573.7045256000101 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9abe0a4e8885b7d6b63c01cf48491534f63f4cec",
          "message": "refactor: generalize SymPy printer implementation (#371)\n\n* fix: instantiate printers in lambdify\r\n* refactor: use _replace_module in _CustomNumPyPrinter init\r\n* refactor: force using _numpycode method",
          "timestamp": "2021-12-05T21:02:34Z",
          "tree_id": "1decf6ce64a303e140413ecc33ceedcbb57a522d",
          "url": "https://github.com/ComPWA/tensorwaves/commit/9abe0a4e8885b7d6b63c01cf48491534f63f4cec"
        },
        "date": 1638738362731,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2699649158724693,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7041850299999624 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4604961721644776,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.1715707110000153 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.371971257677124,
            "unit": "iter/sec",
            "range": "stddev: 0.0004627277322573624",
            "extra": "mean: 54.43074049999552 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 160.34819648462232,
            "unit": "iter/sec",
            "range": "stddev: 0.0003411161306216072",
            "extra": "mean: 6.23642811034611 msec\nrounds: 145"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5686297333641717,
            "unit": "iter/sec",
            "range": "stddev: 0.11062079502513825",
            "extra": "mean: 280.2196009999875 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 99.27031876430608,
            "unit": "iter/sec",
            "range": "stddev: 0.0005501316849566911",
            "extra": "mean: 10.073504471908302 msec\nrounds: 89"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.6607235192394,
            "unit": "iter/sec",
            "range": "stddev: 0.0007066106446124896",
            "extra": "mean: 130.53597319999426 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.41987335946607,
            "unit": "iter/sec",
            "range": "stddev: 0.000453849271261729",
            "extra": "mean: 106.15853970001581 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.390531401917984,
            "unit": "iter/sec",
            "range": "stddev: 0.0014493543832494364",
            "extra": "mean: 106.49024609999742 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.517323615403969,
            "unit": "iter/sec",
            "range": "stddev: 0.001817292745984741",
            "extra": "mean: 659.0551876000177 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.784225784417302,
            "unit": "iter/sec",
            "range": "stddev: 0.00048563391647331755",
            "extra": "mean: 147.40075460001663 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.826651259857211,
            "unit": "iter/sec",
            "range": "stddev: 0.0006191916926080635",
            "extra": "mean: 113.2932491111218 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.797097557574128,
            "unit": "iter/sec",
            "range": "stddev: 0.0010079851592854597",
            "extra": "mean: 113.67385588886867 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.7184062793281807,
            "unit": "iter/sec",
            "range": "stddev: 0.0022314452700474344",
            "extra": "mean: 581.9345588000033 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7e2c2afe90f5c0d05f4d151bbb5c2656638d3e16",
          "message": "feat: add option to lambdify with SymPy's cse (#374)\n\nSee cse argument in\r\nhttps://docs.sympy.org/latest/modules/utilities/lambdify.html#sympy.utilities.lambdify.lambdify",
          "timestamp": "2021-12-06T12:24:35+01:00",
          "tree_id": "58836cd3d80c381d2d06fb2203df00b9f3218c1b",
          "url": "https://github.com/ComPWA/tensorwaves/commit/7e2c2afe90f5c0d05f4d151bbb5c2656638d3e16"
        },
        "date": 1638790109847,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.24784833614648258,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.034725492000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5081407535476442,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9679586670000049 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.059048357208908,
            "unit": "iter/sec",
            "range": "stddev: 0.004240710110198803",
            "extra": "mean: 66.40525857142165 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 111.35662943578241,
            "unit": "iter/sec",
            "range": "stddev: 0.0007274029477607467",
            "extra": "mean: 8.980156862386751 msec\nrounds: 109"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.71539388783602,
            "unit": "iter/sec",
            "range": "stddev: 0.002830837766381888",
            "extra": "mean: 269.1504669999972 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 53.89726035666661,
            "unit": "iter/sec",
            "range": "stddev: 0.03348529458850789",
            "extra": "mean: 18.553818754097566 msec\nrounds: 61"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.4691741669294185,
            "unit": "iter/sec",
            "range": "stddev: 0.003963742369699059",
            "extra": "mean: 182.842961200015 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 5.715235613659883,
            "unit": "iter/sec",
            "range": "stddev: 0.009461343405670118",
            "extra": "mean: 174.97091416667368 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.892699985074592,
            "unit": "iter/sec",
            "range": "stddev: 0.004928673896803318",
            "extra": "mean: 169.70149550000238 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0333613118801883,
            "unit": "iter/sec",
            "range": "stddev: 0.009738077765901338",
            "extra": "mean: 967.715733600005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.171655938642702,
            "unit": "iter/sec",
            "range": "stddev: 0.004155445011727229",
            "extra": "mean: 193.36166439998124 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.874669052599393,
            "unit": "iter/sec",
            "range": "stddev: 0.006461668885738819",
            "extra": "mean: 170.2223548333374 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.938210896629054,
            "unit": "iter/sec",
            "range": "stddev: 0.009107180879258308",
            "extra": "mean: 168.40088999999483 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1790652558830195,
            "unit": "iter/sec",
            "range": "stddev: 0.010897282895913452",
            "extra": "mean: 848.1294779999985 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "acf5770b68edc4335a635d6c0a3394daaf792950",
          "message": "fix: forward use_cse arguments (#375)\n\n* refactor: enforce specifying use_cse in hidden functions\r\n* refactor: remove `**kwargs` from lambdify functions",
          "timestamp": "2021-12-06T12:05:34Z",
          "tree_id": "a577850100d0e33c64d24ba1e7938bdf5a2c8f11",
          "url": "https://github.com/ComPWA/tensorwaves/commit/acf5770b68edc4335a635d6c0a3394daaf792950"
        },
        "date": 1638792539756,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2620717126613757,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8157494750000183 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5810291440715039,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.7210840630000064 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.91444916922989,
            "unit": "iter/sec",
            "range": "stddev: 0.0027741239406042805",
            "extra": "mean: 55.8208622857137 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 125.09303089821745,
            "unit": "iter/sec",
            "range": "stddev: 0.00038562786974955215",
            "extra": "mean: 7.994050450449592 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.164315431709165,
            "unit": "iter/sec",
            "range": "stddev: 0.008205175337361672",
            "extra": "mean: 240.13550760000157 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 66.92046081192463,
            "unit": "iter/sec",
            "range": "stddev: 0.029564788703344547",
            "extra": "mean: 14.943112881580888 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.152727539006408,
            "unit": "iter/sec",
            "range": "stddev: 0.0038382202620935563",
            "extra": "mean: 139.8068072000001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.761776931936712,
            "unit": "iter/sec",
            "range": "stddev: 0.003751058287459402",
            "extra": "mean: 114.13209988889308 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.669952850455093,
            "unit": "iter/sec",
            "range": "stddev: 0.0020550442851923712",
            "extra": "mean: 115.34088100000555 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4321045528459424,
            "unit": "iter/sec",
            "range": "stddev: 0.011914265671982994",
            "extra": "mean: 698.2730402000016 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.5735314185864375,
            "unit": "iter/sec",
            "range": "stddev: 0.004128134337120862",
            "extra": "mean: 152.12523320000173 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.365730220055593,
            "unit": "iter/sec",
            "range": "stddev: 0.002911218482836564",
            "extra": "mean: 119.53529144445143 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.14519610651077,
            "unit": "iter/sec",
            "range": "stddev: 0.0031240887778590343",
            "extra": "mean: 122.77175244444531 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6704955677952285,
            "unit": "iter/sec",
            "range": "stddev: 0.011931955641234808",
            "extra": "mean: 598.6247549999973 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b7a4efd24ef78f4efbf52f48af5b8b1a0db0ac77",
          "message": "refactor!: adapt implementation to AmpForm v0.12.x (#345)\n\n* build: switch to AmpForm v0.12\r\n* ci: update pip constraints and pre-commit config\r\n* feat: compute kinematic helicity angles with different backends\r\n* feat: define PositionalArgumentFunction\r\n* feat: define create_function\r\n* fix: force-push to matching branches\r\n* refactor: accept only str as DataSample keys\r\n* refactor: extract _printer module from sympy module\r\n* refactor: implement SympyDataTransformer\r\n* test: benchmark data generation with numpy and jax\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2021-12-06T12:57:57Z",
          "tree_id": "bcebee9f642abdb302a6cc709100547a10d2a639",
          "url": "https://github.com/ComPWA/tensorwaves/commit/b7a4efd24ef78f4efbf52f48af5b8b1a0db0ac77"
        },
        "date": 1638795743546,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.18273220568632734,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.472489078999956 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.20244738571712517,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.9395550179999645 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.1667861101780995,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.995703113000047 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.32910698097646623,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.0385256400000458 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.561364070996156,
            "unit": "iter/sec",
            "range": "stddev: 0.0004898773142628424",
            "extra": "mean: 64.26171866667119 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 118.71788698857253,
            "unit": "iter/sec",
            "range": "stddev: 0.00015151919959075663",
            "extra": "mean: 8.423330513760385 msec\nrounds: 109"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.1082592362927004,
            "unit": "iter/sec",
            "range": "stddev: 0.11549459483064546",
            "extra": "mean: 321.72348699998565 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 71.84030511148806,
            "unit": "iter/sec",
            "range": "stddev: 0.00026054845195279494",
            "extra": "mean: 13.919762707690518 msec\nrounds: 65"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.1211906526760975,
            "unit": "iter/sec",
            "range": "stddev: 0.0027622150796428",
            "extra": "mean: 163.3669095999835 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 7.8743010286873645,
            "unit": "iter/sec",
            "range": "stddev: 0.00038434036553257843",
            "extra": "mean: 126.99539887500322 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 7.719050334770303,
            "unit": "iter/sec",
            "range": "stddev: 0.0037392879942421634",
            "extra": "mean: 129.54961512500063 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.1326939335280648,
            "unit": "iter/sec",
            "range": "stddev: 0.0014730595633193698",
            "extra": "mean: 882.8510248000043 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.938868435921374,
            "unit": "iter/sec",
            "range": "stddev: 0.00020811446372091954",
            "extra": "mean: 168.3822449999866 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.41535444963582,
            "unit": "iter/sec",
            "range": "stddev: 0.00041079029671448633",
            "extra": "mean: 134.85532037502423 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.314116414642832,
            "unit": "iter/sec",
            "range": "stddev: 0.0003003894119382458",
            "extra": "mean: 136.72191462498517 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2980816626227387,
            "unit": "iter/sec",
            "range": "stddev: 0.0009539482474648452",
            "extra": "mean: 770.3675575999796 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "563db33742da25823ffa92cd4ca361299d748778",
          "message": "feat: implement get_source_code function (#378)\n\n* chore: outsource argument order to PositionalArgumentFunction\r\n* feat: define get_source_code function\r\n* fix: do not replace variables with dummies if use_cse=False\r\n* fix: match argument names in Function interface\r\n* fix: max ParametrizedBackendFunction and PositionalArgumentFunction signatures\r\n* fix: remove faulty complex_twice argument\r\n* refactor: expose function and argument order\r\n* style: avoid word \"dataset\"",
          "timestamp": "2021-12-06T16:06:25Z",
          "tree_id": "5be094725dba5b271d79cf76d692e2881e023baa",
          "url": "https://github.com/ComPWA/tensorwaves/commit/563db33742da25823ffa92cd4ca361299d748778"
        },
        "date": 1638807005319,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2347563223974117,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.259736180000004 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25875308392728463,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.864688237999985 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21690272685329834,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.610361587 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3902848458704863,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.562231177000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 20.088728851805776,
            "unit": "iter/sec",
            "range": "stddev: 0.0011173440581714684",
            "extra": "mean: 49.779157625003734 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.82487808988975,
            "unit": "iter/sec",
            "range": "stddev: 0.00010316176368825911",
            "extra": "mean: 7.308612395349847 msec\nrounds: 129"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.529763293042943,
            "unit": "iter/sec",
            "range": "stddev: 0.0008909210965823594",
            "extra": "mean: 220.7620873999872 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 87.08463543810889,
            "unit": "iter/sec",
            "range": "stddev: 0.0001488884774464086",
            "extra": "mean: 11.48308188889073 msec\nrounds: 81"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.411412117280189,
            "unit": "iter/sec",
            "range": "stddev: 0.0014346568991398328",
            "extra": "mean: 134.92705360000627 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.906234607625414,
            "unit": "iter/sec",
            "range": "stddev: 0.0009756293849301382",
            "extra": "mean: 100.94652909090614 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.04418929208615,
            "unit": "iter/sec",
            "range": "stddev: 0.0010674966814439434",
            "extra": "mean: 99.56005118182145 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4861134593561527,
            "unit": "iter/sec",
            "range": "stddev: 0.0017547465901802473",
            "extra": "mean: 672.8961329999947 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.628999647635787,
            "unit": "iter/sec",
            "range": "stddev: 0.0003033547528817439",
            "extra": "mean: 131.0787843999833 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.170553884103441,
            "unit": "iter/sec",
            "range": "stddev: 0.00020426718364729754",
            "extra": "mean: 109.0446676000056 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.255232333379245,
            "unit": "iter/sec",
            "range": "stddev: 0.0008543183907625174",
            "extra": "mean: 108.04699050000863 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.7002361426694106,
            "unit": "iter/sec",
            "range": "stddev: 0.004011493950945459",
            "extra": "mean: 588.1535952000036 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "15ef2aec870029abc7c86b056c50050f474f02fc",
          "message": "build: make ampform an optional dependency (#380)\n\n* docs: explain optional dependency syntax",
          "timestamp": "2021-12-07T14:56:41Z",
          "tree_id": "394c126d2dd70d82ff29a16e50a94663cb0095ae",
          "url": "https://github.com/ComPWA/tensorwaves/commit/15ef2aec870029abc7c86b056c50050f474f02fc"
        },
        "date": 1638889335127,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.14607386276756315,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.845851687999982 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.17086905605687447,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.852434742000014 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.132370500843658,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 7.554553270000042 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.26962895212357163,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7088005280000402 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 11.553031857876572,
            "unit": "iter/sec",
            "range": "stddev: 0.006484352421469091",
            "extra": "mean: 86.55736539999452 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 81.36606382523999,
            "unit": "iter/sec",
            "range": "stddev: 0.0019332168245178782",
            "extra": "mean: 12.290136120481682 msec\nrounds: 83"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.7030514825493874,
            "unit": "iter/sec",
            "range": "stddev: 0.13336556679308906",
            "extra": "mean: 369.9522581999986 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 47.64241116081056,
            "unit": "iter/sec",
            "range": "stddev: 0.0032780197075985983",
            "extra": "mean: 20.989701730767454 msec\nrounds: 52"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.099906531462792,
            "unit": "iter/sec",
            "range": "stddev: 0.011442061121628938",
            "extra": "mean: 196.0820250000097 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.532635296861203,
            "unit": "iter/sec",
            "range": "stddev: 0.009085503199406594",
            "extra": "mean: 220.62220639999168 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.380807479862421,
            "unit": "iter/sec",
            "range": "stddev: 0.007977013740388946",
            "extra": "mean: 228.26841959998774 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.6933020914549778,
            "unit": "iter/sec",
            "range": "stddev: 0.04361775795963705",
            "extra": "mean: 1.4423726862000081 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.602032773100111,
            "unit": "iter/sec",
            "range": "stddev: 0.012295843857043636",
            "extra": "mean: 217.29528000000755 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.30421200821174,
            "unit": "iter/sec",
            "range": "stddev: 0.00829265088546602",
            "extra": "mean: 232.3305631999915 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.464592041106716,
            "unit": "iter/sec",
            "range": "stddev: 0.01065211706343402",
            "extra": "mean: 223.98463080002102 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.7908371066955625,
            "unit": "iter/sec",
            "range": "stddev: 0.03635075968288968",
            "extra": "mean: 1.2644829023999704 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0e5ed893109c2d58c963ad3ffb3588962918a88f",
          "message": "docs: add GPU installation tips (#381)\n\n* ci: check anchors with linkcheck\r\n* docs: add GPU installation instructions\r\n* fix: remove typing-extensions",
          "timestamp": "2021-12-08T17:19:23+01:00",
          "tree_id": "09aa339b04e32150b86bfc151d0cb1c582f47498",
          "url": "https://github.com/ComPWA/tensorwaves/commit/0e5ed893109c2d58c963ad3ffb3588962918a88f"
        },
        "date": 1638980598551,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22085755024132303,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.527805361000048 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2531798835107291,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.949760882000021 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20854319125907309,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.795169739000016 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.40973242694691125,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4406171790000144 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.488495012335623,
            "unit": "iter/sec",
            "range": "stddev: 0.0006033977775685644",
            "extra": "mean: 54.08769071429528 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.65444520818775,
            "unit": "iter/sec",
            "range": "stddev: 0.00012003697771417708",
            "extra": "mean: 7.212174110238685 msec\nrounds: 127"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.689770360072208,
            "unit": "iter/sec",
            "range": "stddev: 0.1020189543101376",
            "extra": "mean: 271.0195763999877 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 84.1906733142357,
            "unit": "iter/sec",
            "range": "stddev: 0.00019620068643858027",
            "extra": "mean: 11.877800243591961 msec\nrounds: 78"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.767754565912201,
            "unit": "iter/sec",
            "range": "stddev: 0.0004996012028502721",
            "extra": "mean: 128.73733220001213 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.56783614965811,
            "unit": "iter/sec",
            "range": "stddev: 0.0004136398924233377",
            "extra": "mean: 104.51683999999659 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.562774569730344,
            "unit": "iter/sec",
            "range": "stddev: 0.0011203741954092228",
            "extra": "mean: 104.5721608000008 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3599518123160605,
            "unit": "iter/sec",
            "range": "stddev: 0.0006458195353909764",
            "extra": "mean: 735.3201716000171 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.087432559624156,
            "unit": "iter/sec",
            "range": "stddev: 0.00011555145436115373",
            "extra": "mean: 141.09481699999833 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.98163789458071,
            "unit": "iter/sec",
            "range": "stddev: 0.0005156791096504739",
            "extra": "mean: 111.33826722221505 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.803605549128314,
            "unit": "iter/sec",
            "range": "stddev: 0.0009622033988003093",
            "extra": "mean: 113.58982344444257 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5674132625945978,
            "unit": "iter/sec",
            "range": "stddev: 0.001851752248633385",
            "extra": "mean: 637.9938359999983 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "Leo-Wol@web.de",
            "name": "Leongrim",
            "username": "Leongrim"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8f9ec18093bf04e6825616dca3c3f1a354e7f2bd",
          "message": "build: set minimal dependencies sympy and pyyaml (#383)\n\n* ci: update pip constraints and pre-commit config\r\n\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2021-12-15T16:13:38+01:00",
          "tree_id": "d26e279b8e8251cbcf7fbe5eb77a9bb1c0d9c2ac",
          "url": "https://github.com/ComPWA/tensorwaves/commit/8f9ec18093bf04e6825616dca3c3f1a354e7f2bd"
        },
        "date": 1639581445540,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2224156879877077,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.49608572599999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.24806850697595123,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.03114450999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21543337669441281,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.641806276000011 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4051037849443757,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4685032259999957 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.527840118869033,
            "unit": "iter/sec",
            "range": "stddev: 0.0008258444056126481",
            "extra": "mean: 53.97283188889269 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.6201852115676,
            "unit": "iter/sec",
            "range": "stddev: 0.00012768289471181058",
            "extra": "mean: 7.2139565999984825 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.603269981056143,
            "unit": "iter/sec",
            "range": "stddev: 0.11771983146969646",
            "extra": "mean: 277.52569339999695 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 85.85920093423525,
            "unit": "iter/sec",
            "range": "stddev: 0.00022637491941673548",
            "extra": "mean: 11.646975386667767 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.634884970700759,
            "unit": "iter/sec",
            "range": "stddev: 0.0004653388066952385",
            "extra": "mean: 130.97774279999612 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.576483640487183,
            "unit": "iter/sec",
            "range": "stddev: 0.00038576271341519253",
            "extra": "mean: 104.42246209999553 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.407940194461819,
            "unit": "iter/sec",
            "range": "stddev: 0.0011833165501323305",
            "extra": "mean: 106.2931926999994 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3881434396667696,
            "unit": "iter/sec",
            "range": "stddev: 0.0029013436890253665",
            "extra": "mean: 720.3866484000059 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.4994960238785575,
            "unit": "iter/sec",
            "range": "stddev: 0.0003680885861795762",
            "extra": "mean: 153.85808320000365 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.901166582243555,
            "unit": "iter/sec",
            "range": "stddev: 0.0002714476624955549",
            "extra": "mean: 112.34482477778191 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.835466212373742,
            "unit": "iter/sec",
            "range": "stddev: 0.0010678924020870691",
            "extra": "mean: 113.18021889999841 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5802236801562484,
            "unit": "iter/sec",
            "range": "stddev: 0.001570096923057975",
            "extra": "mean: 632.8218040000024 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "30473cae2376430df95680490e08cd4f5476051f",
          "message": "ci: update pip constraints and pre-commit config (#385)\n\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2021-12-20T10:13:38Z",
          "tree_id": "9033a745ab5b08fdc8b50f5c1284e939d4332e27",
          "url": "https://github.com/ComPWA/tensorwaves/commit/30473cae2376430df95680490e08cd4f5476051f"
        },
        "date": 1639995496055,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.16476588930670433,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.069217385999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.1879488447592988,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.3205966829999625 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.15786458356004937,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.334543046000022 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3333317828961009,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.0000139539999964 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.03666050072197,
            "unit": "iter/sec",
            "range": "stddev: 0.004693067351676189",
            "extra": "mean: 71.24201657143203 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 108.68525035453258,
            "unit": "iter/sec",
            "range": "stddev: 0.0013079546979771364",
            "extra": "mean: 9.200880494252791 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.714924898532169,
            "unit": "iter/sec",
            "range": "stddev: 0.011296240176102295",
            "extra": "mean: 269.18444579999914 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 58.70398982008111,
            "unit": "iter/sec",
            "range": "stddev: 0.0017503145979526839",
            "extra": "mean: 17.03461729032131 msec\nrounds: 62"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.9660809223877465,
            "unit": "iter/sec",
            "range": "stddev: 0.015235415029047553",
            "extra": "mean: 201.36603000000832 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 5.661071160145349,
            "unit": "iter/sec",
            "range": "stddev: 0.010745464356319666",
            "extra": "mean: 176.64501500001015 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.377027064821254,
            "unit": "iter/sec",
            "range": "stddev: 0.012693118768027225",
            "extra": "mean: 185.97637466666583 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.7869956560767337,
            "unit": "iter/sec",
            "range": "stddev: 0.047210895056712894",
            "extra": "mean: 1.2706550440000115 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.59922970500349,
            "unit": "iter/sec",
            "range": "stddev: 0.02215561693053349",
            "extra": "mean: 217.42771380000931 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.601632353011528,
            "unit": "iter/sec",
            "range": "stddev: 0.006922926271677092",
            "extra": "mean: 178.519391666678 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.384361347961842,
            "unit": "iter/sec",
            "range": "stddev: 0.010580462857916719",
            "extra": "mean: 185.72304780000195 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9317353100779353,
            "unit": "iter/sec",
            "range": "stddev: 0.04332630236697684",
            "extra": "mean: 1.0732661831999848 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8654dbfaef8866179a47f936056a2cd90d368fd1",
          "message": "ci: update pip constraints and pre-commit config (#386)\n\n* ci: update pip constraints and pre-commit config\r\n* ci: remove specific language_info from notebooks\r\n* fix: remove black, mypy and pylint from Python 3.6\r\n  These cause too many issues and probably no one will be developing in\r\n  Python 3.6.\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2021-12-27T22:22:41Z",
          "tree_id": "df0958fc8cb79d5097fbf44d57cec7350ed2d456",
          "url": "https://github.com/ComPWA/tensorwaves/commit/8654dbfaef8866179a47f936056a2cd90d368fd1"
        },
        "date": 1640643984053,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21520807369254927,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.646665818999992 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25275273439022944,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9564359309999872 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2045972998964928,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.887650035000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.40551055790978835,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.466027037999993 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.374589489469045,
            "unit": "iter/sec",
            "range": "stddev: 0.0006555566491398223",
            "extra": "mean: 54.42298455555298 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.9283851578807,
            "unit": "iter/sec",
            "range": "stddev: 0.00012153566746863896",
            "extra": "mean: 7.250139257813704 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.469815154995783,
            "unit": "iter/sec",
            "range": "stddev: 0.000870271578989664",
            "extra": "mean: 223.72289800000544 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 84.41268230784581,
            "unit": "iter/sec",
            "range": "stddev: 0.00022638022155729813",
            "extra": "mean: 11.846561116883905 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.578966544642429,
            "unit": "iter/sec",
            "range": "stddev: 0.000514594736447887",
            "extra": "mean: 131.94411059999993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.400341066370938,
            "unit": "iter/sec",
            "range": "stddev: 0.00019709495516804712",
            "extra": "mean: 106.37911889999714 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.410527547003994,
            "unit": "iter/sec",
            "range": "stddev: 0.0010432172677430475",
            "extra": "mean: 106.263968199994 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3637047699190044,
            "unit": "iter/sec",
            "range": "stddev: 0.0009540391935679579",
            "extra": "mean: 733.2965478000006 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.243438629866686,
            "unit": "iter/sec",
            "range": "stddev: 0.00017153209286186277",
            "extra": "mean: 160.16814759999534 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.820236169056345,
            "unit": "iter/sec",
            "range": "stddev: 0.0005228694003462826",
            "extra": "mean: 113.37564899999582 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.725469488962082,
            "unit": "iter/sec",
            "range": "stddev: 0.0010083479012163669",
            "extra": "mean: 114.60701355554824 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5612660460620782,
            "unit": "iter/sec",
            "range": "stddev: 0.0016367607783904844",
            "extra": "mean: 640.5058269999927 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ad7a1bf76f607bd2f559eb2947b7816ac0005e30",
          "message": "feat!: implement chi-squared estimator (#387)\n\n* docs: explain mathematics of UnbinnedNLL estimator\r\n* docs: illustrate chi-squared fit example\r\n* docs: overwrite docstring of Estimator.__call__",
          "timestamp": "2021-12-28T12:47:58Z",
          "tree_id": "a160c73bb175775671c73d0adea86685235b558f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/ad7a1bf76f607bd2f559eb2947b7816ac0005e30"
        },
        "date": 1640695908319,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21334474940518702,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.687249171999952 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.23980161103369912,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.170113768999954 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.19630875994019603,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.094016182999894 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.39304289193793857,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5442515830000048 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.63720180642854,
            "unit": "iter/sec",
            "range": "stddev: 0.0011405783207088316",
            "extra": "mean: 56.698336333346965 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.515780031293,
            "unit": "iter/sec",
            "range": "stddev: 0.00013667790185049994",
            "extra": "mean: 7.271892722220248 msec\nrounds: 126"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.359560579695468,
            "unit": "iter/sec",
            "range": "stddev: 0.0006863920530052168",
            "extra": "mean: 229.3809162000116 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 83.80421926465996,
            "unit": "iter/sec",
            "range": "stddev: 0.00016678633724363716",
            "extra": "mean: 11.932573428575543 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.508404153014514,
            "unit": "iter/sec",
            "range": "stddev: 0.0005092058278299186",
            "extra": "mean: 133.18409339999562 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.553676993117767,
            "unit": "iter/sec",
            "range": "stddev: 0.0005680948185516821",
            "extra": "mean: 104.6717406000198 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.307325649894358,
            "unit": "iter/sec",
            "range": "stddev: 0.0009386938936336053",
            "extra": "mean: 107.44224899999608 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3654594052462752,
            "unit": "iter/sec",
            "range": "stddev: 0.0009749713003112446",
            "extra": "mean: 732.3542509999697 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.362056191977451,
            "unit": "iter/sec",
            "range": "stddev: 0.0003273988510435047",
            "extra": "mean: 157.18188740002006 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.731865853822956,
            "unit": "iter/sec",
            "range": "stddev: 0.00034747109404987197",
            "extra": "mean: 114.52306033334025 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.744130783966083,
            "unit": "iter/sec",
            "range": "stddev: 0.0003758425210140363",
            "extra": "mean: 114.36242488889548 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5584871498174384,
            "unit": "iter/sec",
            "range": "stddev: 0.0018115381284238747",
            "extra": "mean: 641.6478955999992 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6b1f04750b693d191790f103bd5855ea1dcc28aa",
          "message": "docs: illustrate binned and unbinned fit (#388)\n\n* docs: illustrate unbinned 2D fit\r\n* docs: illustrate binned fit with ChiSquared estimator",
          "timestamp": "2021-12-28T14:04:05+01:00",
          "tree_id": "4456508f4e8c666a35af712e566bdd12a13fa20f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/6b1f04750b693d191790f103bd5855ea1dcc28aa"
        },
        "date": 1640696873901,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22647631466493198,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.415472767999972 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.23677238211471077,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.223465553999972 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.18795156226236148,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.320519754999964 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.40982652585360924,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.440056797000011 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.50447764538446,
            "unit": "iter/sec",
            "range": "stddev: 0.00028915819211384314",
            "extra": "mean: 57.12823999999096 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.42590462471668,
            "unit": "iter/sec",
            "range": "stddev: 0.00013820663277591206",
            "extra": "mean: 7.276648479999494 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5186274027040954,
            "unit": "iter/sec",
            "range": "stddev: 0.11889743921875305",
            "extra": "mean: 284.20173140000315 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.89744557721671,
            "unit": "iter/sec",
            "range": "stddev: 0.0002641172366433646",
            "extra": "mean: 12.063097880000743 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.398702335381935,
            "unit": "iter/sec",
            "range": "stddev: 0.0006786054199165644",
            "extra": "mean: 135.1588366000101 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.41474954035393,
            "unit": "iter/sec",
            "range": "stddev: 0.00032471163858231934",
            "extra": "mean: 106.2163147000092 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.266823381067658,
            "unit": "iter/sec",
            "range": "stddev: 0.00044836641676973917",
            "extra": "mean: 107.91184409999914 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.346166701269182,
            "unit": "iter/sec",
            "range": "stddev: 0.0008404563612029081",
            "extra": "mean: 742.8500490000147 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.1812327350122835,
            "unit": "iter/sec",
            "range": "stddev: 0.000021959205834703476",
            "extra": "mean: 161.7800272000295 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.788497667374497,
            "unit": "iter/sec",
            "range": "stddev: 0.00030418683064659786",
            "extra": "mean: 113.7850902222226 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.734684364232214,
            "unit": "iter/sec",
            "range": "stddev: 0.0009677232147694047",
            "extra": "mean: 114.48610599999635 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5179040354122735,
            "unit": "iter/sec",
            "range": "stddev: 0.0006938524110052572",
            "extra": "mean: 658.8031764000107 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e55fa215f1edf130fb5a49717af1382e03dd94b0",
          "message": "docs!: merge and isolate AmpForm notebooks (#389)\n\n* ci: update pip constraints and pre-commit config\r\n* docs: generalize main usage page\r\n* docs: illustrate loadable callbacks in basics\r\n* docs: improve index page buttons\r\n* docs: link to estimator and optimizer implementations\r\n* docs: link to the three ComPWA packages\r\n* docs: link to use_cse PR in faster-lambdify\r\n* docs: move scipy miminizer example to basics notebook\r\n* docs: simplify amplitude analysis notebook\r\n* fix: use JAX functions in fit examples (faster)\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2021-12-28T17:20:27+01:00",
          "tree_id": "72dc0df3481a4636b2e5bd588a6806de2f01a7f9",
          "url": "https://github.com/ComPWA/tensorwaves/commit/e55fa215f1edf130fb5a49717af1382e03dd94b0"
        },
        "date": 1640708698800,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.17080143669949388,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.854751689000068 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.21403337630391314,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.672168506000048 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.16303765016993818,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.133552581000004 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.35664822910162947,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.8038832619999994 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.20218181887374,
            "unit": "iter/sec",
            "range": "stddev: 0.004396290334920291",
            "extra": "mean: 65.7800315714212 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 117.89974217650143,
            "unit": "iter/sec",
            "range": "stddev: 0.0008456721701490161",
            "extra": "mean: 8.481782754901644 msec\nrounds: 102"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.009113124665456,
            "unit": "iter/sec",
            "range": "stddev: 0.006396673884393913",
            "extra": "mean: 249.4317244000058 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 63.15597352430837,
            "unit": "iter/sec",
            "range": "stddev: 0.0015808935299583852",
            "extra": "mean: 15.833814985293602 msec\nrounds: 68"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.456941843767708,
            "unit": "iter/sec",
            "range": "stddev: 0.005665276804564114",
            "extra": "mean: 183.25282339999376 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 5.821465770832018,
            "unit": "iter/sec",
            "range": "stddev: 0.011196911594388625",
            "extra": "mean: 171.77804342858443 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.813956868578834,
            "unit": "iter/sec",
            "range": "stddev: 0.007550726160324084",
            "extra": "mean: 171.99990000002194 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8565869392188733,
            "unit": "iter/sec",
            "range": "stddev: 0.03561711135211002",
            "extra": "mean: 1.1674238237999588 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.165009478811865,
            "unit": "iter/sec",
            "range": "stddev: 0.0059931336056863955",
            "extra": "mean: 240.09549200000038 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.6203131997474784,
            "unit": "iter/sec",
            "range": "stddev: 0.011591484902824408",
            "extra": "mean: 177.92602733330418 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.828099720830364,
            "unit": "iter/sec",
            "range": "stddev: 0.0066693156977873265",
            "extra": "mean: 171.5825136666543 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9913282803286164,
            "unit": "iter/sec",
            "range": "stddev: 0.02225562873828821",
            "extra": "mean: 1.008747576199994 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "17f015b9b1908095244f47eb5b7a8415af03dffa",
          "message": "docs: illustrate expression tree optimization (#390)\n\n* docs: use create_function in sub-intensities example\r\n* docs: set more free parameters\r\n* docs: move AIC etc. to \"Analyze fit result\" section",
          "timestamp": "2021-12-28T18:19:41Z",
          "tree_id": "3398bd6d2148f754560df9cfaee29dc35e29100d",
          "url": "https://github.com/ComPWA/tensorwaves/commit/17f015b9b1908095244f47eb5b7a8415af03dffa"
        },
        "date": 1640715803596,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2176415926051424,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.594709991000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25336843150610605,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.946821607000004 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20548710865928313,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.8664853309999785 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4152396404499882,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4082479190000186 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.175916644129828,
            "unit": "iter/sec",
            "range": "stddev: 0.000807711731003849",
            "extra": "mean: 52.14874566667049 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.30618169897224,
            "unit": "iter/sec",
            "range": "stddev: 0.00010519410566556715",
            "extra": "mean: 7.2829932900791245 msec\nrounds: 131"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.487863784566971,
            "unit": "iter/sec",
            "range": "stddev: 0.002613564703935524",
            "extra": "mean: 222.82316219998393 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 85.20465770755216,
            "unit": "iter/sec",
            "range": "stddev: 0.00014236067716040255",
            "extra": "mean: 11.736447594593933 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.608862456634663,
            "unit": "iter/sec",
            "range": "stddev: 0.0005239421072152442",
            "extra": "mean: 131.42569019998973 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.339358056085503,
            "unit": "iter/sec",
            "range": "stddev: 0.00030668717537891396",
            "extra": "mean: 107.07374040000559 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.547040564927778,
            "unit": "iter/sec",
            "range": "stddev: 0.001224583515778413",
            "extra": "mean: 104.74450099998762 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3637515579947888,
            "unit": "iter/sec",
            "range": "stddev: 0.002411007374456325",
            "extra": "mean: 733.2713895999973 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.342726209344164,
            "unit": "iter/sec",
            "range": "stddev: 0.00014909143783164378",
            "extra": "mean: 157.66091219999225 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.859665686324295,
            "unit": "iter/sec",
            "range": "stddev: 0.0003572750142496766",
            "extra": "mean: 112.87107611109883 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.76086780297137,
            "unit": "iter/sec",
            "range": "stddev: 0.0002713169322652091",
            "extra": "mean: 114.14394355555008 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.552866614348774,
            "unit": "iter/sec",
            "range": "stddev: 0.00044108018872844915",
            "extra": "mean: 643.9703131999977 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c90b0aa83312b50e253dfc459815fe591c230b0d",
          "message": "ci: update pip constraints and pre-commit config (#391)\n\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-01-03T10:04:31+01:00",
          "tree_id": "de0269a80d4b19bc0418d2672d43a058d0306638",
          "url": "https://github.com/ComPWA/tensorwaves/commit/c90b0aa83312b50e253dfc459815fe591c230b0d"
        },
        "date": 1641200887800,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.25411051126018214,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.935295690999993 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26473256646907606,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7773969910000176 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21493377003083766,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.65259600600001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.42233746198230687,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.367774800999996 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 20.392237113083922,
            "unit": "iter/sec",
            "range": "stddev: 0.0034968387063847345",
            "extra": "mean: 49.03826855555672 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 156.95986254997618,
            "unit": "iter/sec",
            "range": "stddev: 0.00010291181598473134",
            "extra": "mean: 6.371055528171089 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.19410583869204,
            "unit": "iter/sec",
            "range": "stddev: 0.09186978721115927",
            "extra": "mean: 238.42984379999734 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 95.71126203971852,
            "unit": "iter/sec",
            "range": "stddev: 0.00013920027653349982",
            "extra": "mean: 10.448091255812898 msec\nrounds: 86"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.292420488575292,
            "unit": "iter/sec",
            "range": "stddev: 0.0006844942004526717",
            "extra": "mean: 120.59205166666705 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.334659596484174,
            "unit": "iter/sec",
            "range": "stddev: 0.0003124778841163674",
            "extra": "mean: 96.76177436363726 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.329284247647644,
            "unit": "iter/sec",
            "range": "stddev: 0.001031903468161372",
            "extra": "mean: 96.8121290909132 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4794562194516006,
            "unit": "iter/sec",
            "range": "stddev: 0.0022027622950961676",
            "extra": "mean: 675.9240232000082 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.007260289655889,
            "unit": "iter/sec",
            "range": "stddev: 0.00012961128588898747",
            "extra": "mean: 142.70912719999842 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.756273887778708,
            "unit": "iter/sec",
            "range": "stddev: 0.000286630596475629",
            "extra": "mean: 102.4981475000061 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.732652446311194,
            "unit": "iter/sec",
            "range": "stddev: 0.0008471327891106271",
            "extra": "mean: 102.74691360000361 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6996654770077817,
            "unit": "iter/sec",
            "range": "stddev: 0.0011736790681531842",
            "extra": "mean: 588.3510688000058 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bc065b7edc6446dd6ec30382c66ff9144b08ee42",
          "message": "refactor!: generalize data generation interface (#392)\n\n* chore: allow importing data classes from main sub-module\r\n* chore: collect RNGs under data.rng module\r\n* chore: increase default bunch size to 50,000\r\n* chore: isolate data sample handling functions\r\n* docs: add docstring to TFPhaseSpaceGenerator.generate\r\n* docs: add note about how to use amplitude analysis examples\r\n* docs: fix y-tick labels in fit animation\r\n* docs: use general data generators in notebooks\r\n* docs: write \"for a Function\" instead of \"with a Function\"\r\n* feat: define generate DataGenerator interface\r\n* feat: implement IdentityTransformer\r\n* feat: implement IntensityDistributionGenerator\r\n* feat: implement NumpyDomainGenerator\r\n* feat: implement NumpyUniformRNG\r\n* feat: implemented unweighted TFPhaseSpaceGenerator\r\n* fix: import NumpyUniformRNG from data.rng\r\n* fix: remove __all__ statement from data module\r\n* fix: use type function instead of __class__\r\n* refactor!: remove generate_phsp and generate_data facade functions\r\n* refactor: allow IntensityDistributionGenerator with WeightedDataGenerator\r\n* refactor: merge PhaseSpaceGenerator.setup() into its constructor\r\n* refactor: rename PhaseSpaceGenerator to WeightedDataGenerator\r\n* refactor: rename UniformRealNumberGenerator to RealNumberGenerator\r\n* style: use pytest.approx instead of numpy testing\r\n* test: merge test_generate with test_data",
          "timestamp": "2022-01-03T16:39:31+01:00",
          "tree_id": "44afd31f240b8c85aadd13d394378d48cbe026bf",
          "url": "https://github.com/ComPWA/tensorwaves/commit/bc065b7edc6446dd6ec30382c66ff9144b08ee42"
        },
        "date": 1641224605288,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.20628077783853196,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.847761437000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.24944386913431066,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.008917932000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.1884336168726118,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.306908696000022 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4079555967584342,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4512471650000123 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.9532625664961,
            "unit": "iter/sec",
            "range": "stddev: 0.001459887435083148",
            "extra": "mean: 52.76136477778297 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 132.2003873814389,
            "unit": "iter/sec",
            "range": "stddev: 0.0004362692063189447",
            "extra": "mean: 7.564274355071982 msec\nrounds: 138"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.529585706072487,
            "unit": "iter/sec",
            "range": "stddev: 0.007249619643608007",
            "extra": "mean: 220.770742600007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 73.96178994517878,
            "unit": "iter/sec",
            "range": "stddev: 0.0006130230227905951",
            "extra": "mean: 13.520494849316249 msec\nrounds: 73"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.704010587839074,
            "unit": "iter/sec",
            "range": "stddev: 0.008003007365857224",
            "extra": "mean: 175.31524260000424 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 6.6353676563894295,
            "unit": "iter/sec",
            "range": "stddev: 0.0019970164543084965",
            "extra": "mean: 150.70754957143404 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 6.681418564352058,
            "unit": "iter/sec",
            "range": "stddev: 0.0023795731859704975",
            "extra": "mean: 149.66881514284782 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9810913071516786,
            "unit": "iter/sec",
            "range": "stddev: 0.004631752976182806",
            "extra": "mean: 1.0192731224000113 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.330295803859581,
            "unit": "iter/sec",
            "range": "stddev: 0.0013598130347582005",
            "extra": "mean: 187.6068489999966 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 6.872682174436909,
            "unit": "iter/sec",
            "range": "stddev: 0.0009602194095072819",
            "extra": "mean: 145.50360028571114 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 6.768946276517929,
            "unit": "iter/sec",
            "range": "stddev: 0.0014806991604197606",
            "extra": "mean: 147.73348157143573 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1406010376545892,
            "unit": "iter/sec",
            "range": "stddev: 0.006862024807591124",
            "extra": "mean: 876.730747199997 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f9756de52c81cc7b58d788f49563023d9c193832",
          "message": "test: increase test coverage (#393)\n\n* test: run _all_unique check PositionalArgumentFunction\r\n* test: run ParametrizedBackendFunction.update_parameter\r\n* test: ignore TYPE_CHECKING in test coverage",
          "timestamp": "2022-01-05T11:11:13Z",
          "tree_id": "702de33aeff68f910376fd7c85ee403122ebdfcb",
          "url": "https://github.com/ComPWA/tensorwaves/commit/f9756de52c81cc7b58d788f49563023d9c193832"
        },
        "date": 1641381327669,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.19169515984225133,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.216615801999978 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.22141236688213933,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.516459554999983 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.17479483080924862,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.7209929800000054 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.33832870123392106,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.9557054910000033 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.842163134582401,
            "unit": "iter/sec",
            "range": "stddev: 0.00029168953946146647",
            "extra": "mean: 63.12269299998974 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 121.61935117958501,
            "unit": "iter/sec",
            "range": "stddev: 0.00023679111584123965",
            "extra": "mean: 8.222375718181432 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.8056799133352897,
            "unit": "iter/sec",
            "range": "stddev: 0.005241107042792947",
            "extra": "mean: 262.7651359999959 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 73.27310329602268,
            "unit": "iter/sec",
            "range": "stddev: 0.0003097413766520606",
            "extra": "mean: 13.647572642856534 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.406595287457568,
            "unit": "iter/sec",
            "range": "stddev: 0.0018489731157913384",
            "extra": "mean: 156.08914799998956 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.07766125606783,
            "unit": "iter/sec",
            "range": "stddev: 0.0014672404412883924",
            "extra": "mean: 123.79820944444948 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 7.836105792548065,
            "unit": "iter/sec",
            "range": "stddev: 0.002257338975034309",
            "extra": "mean: 127.61440777777328 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.1581618662182078,
            "unit": "iter/sec",
            "range": "stddev: 0.004136173792726797",
            "extra": "mean: 863.4371664000128 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.540190618657004,
            "unit": "iter/sec",
            "range": "stddev: 0.0016580825552927633",
            "extra": "mean: 180.4992046000052 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.5254613330976,
            "unit": "iter/sec",
            "range": "stddev: 0.0011358265061800223",
            "extra": "mean: 132.88221887499674 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.349044838533551,
            "unit": "iter/sec",
            "range": "stddev: 0.00093203182048128",
            "extra": "mean: 136.072104874998 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.3421873783936593,
            "unit": "iter/sec",
            "range": "stddev: 0.004615440936720601",
            "extra": "mean: 745.0524539999833 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5b6333413a72a57e2121ef0449c5053dc2cbac25",
          "message": "build: make tensorflow an optional dependency (#394)\n\n* ci: allow running both benchmarks and tests\r\n* ci: test framework with JAX only\r\n* ci: update pip constraints and pre-commit config\r\n* docs: recommend installing JAX\r\n* feat: provide install instructions on ImportError\r\n* refactor: remove phasespace dependency from rng\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-01-05T11:51:45Z",
          "tree_id": "dc3288964733b483abe93d7936a3bd32dac7c954",
          "url": "https://github.com/ComPWA/tensorwaves/commit/5b6333413a72a57e2121ef0449c5053dc2cbac25"
        },
        "date": 1641383788455,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.1644654447081482,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.080304600000005 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19742668900711516,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.065171305000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.15640664609489502,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.393590201999984 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3577651606352502,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.7951296269999943 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.414274493924554,
            "unit": "iter/sec",
            "range": "stddev: 0.0018631100867196994",
            "extra": "mean: 69.37567342855085 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 103.13332721215934,
            "unit": "iter/sec",
            "range": "stddev: 0.0003784384716406208",
            "extra": "mean: 9.69618674226289 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.000335510118241,
            "unit": "iter/sec",
            "range": "stddev: 0.10602612412386572",
            "extra": "mean: 333.2960585999899 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 61.44351699504254,
            "unit": "iter/sec",
            "range": "stddev: 0.0003752434663746811",
            "extra": "mean: 16.27511003448392 msec\nrounds: 58"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.110347613366912,
            "unit": "iter/sec",
            "range": "stddev: 0.003546753217487677",
            "extra": "mean: 195.68140480000693 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 5.176981573370339,
            "unit": "iter/sec",
            "range": "stddev: 0.003446535442032204",
            "extra": "mean: 193.16275049999376 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.038069969161385,
            "unit": "iter/sec",
            "range": "stddev: 0.0033567063341967945",
            "extra": "mean: 198.48870819998865 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8385411143065201,
            "unit": "iter/sec",
            "range": "stddev: 0.003364070031003099",
            "extra": "mean: 1.1925473694000175 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.544474116477253,
            "unit": "iter/sec",
            "range": "stddev: 0.0011388916203617352",
            "extra": "mean: 220.04746300000306 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.019605739233057,
            "unit": "iter/sec",
            "range": "stddev: 0.003623770887948179",
            "extra": "mean: 199.21883350001698 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.1282107508279084,
            "unit": "iter/sec",
            "range": "stddev: 0.00197451392989655",
            "extra": "mean: 194.9997862000032 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.969982935621765,
            "unit": "iter/sec",
            "range": "stddev: 0.0019709373112797397",
            "extra": "mean: 1.0309459715999991 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "faa0cbf67b0478774e09b649a69ca0d8374cc33a",
          "message": "fix: hide progress bar of domain generator (#396)\n\n* fix: finalize progress bar if total is unspecified",
          "timestamp": "2022-01-05T19:49:02+01:00",
          "tree_id": "2eec979ffe398e136f33e20d7aaa60f5a7175766",
          "url": "https://github.com/ComPWA/tensorwaves/commit/faa0cbf67b0478774e09b649a69ca0d8374cc33a"
        },
        "date": 1641408769981,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22469754177316634,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.450426969999995 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.24708462967602013,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.0471963040000105 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20618597568366723,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.849990386999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41375416199087084,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4168941169999982 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 20.016825876943155,
            "unit": "iter/sec",
            "range": "stddev: 0.0009725711243642297",
            "extra": "mean: 49.957970666661645 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 124.45145269072071,
            "unit": "iter/sec",
            "range": "stddev: 0.018567235745909825",
            "extra": "mean: 8.035261769785365 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.9073396501872266,
            "unit": "iter/sec",
            "range": "stddev: 0.12028966839475237",
            "extra": "mean: 255.92860859999288 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 94.04352737712573,
            "unit": "iter/sec",
            "range": "stddev: 0.00018905518816811367",
            "extra": "mean: 10.633374011907074 msec\nrounds: 84"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.089270732666973,
            "unit": "iter/sec",
            "range": "stddev: 0.0016782756271614301",
            "extra": "mean: 123.62053800000676 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.214894135999852,
            "unit": "iter/sec",
            "range": "stddev: 0.0002461450763019872",
            "extra": "mean: 97.8962666363569 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.23979920163359,
            "unit": "iter/sec",
            "range": "stddev: 0.001104151983468737",
            "extra": "mean: 97.65816499999987 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4787772548792457,
            "unit": "iter/sec",
            "range": "stddev: 0.0005868496247247288",
            "extra": "mean: 676.2343664000014 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.875965339980134,
            "unit": "iter/sec",
            "range": "stddev: 0.0001493122596067611",
            "extra": "mean: 145.43412460000695 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.642566797939642,
            "unit": "iter/sec",
            "range": "stddev: 0.00021154106367589787",
            "extra": "mean: 103.70682629999237 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.534728649656033,
            "unit": "iter/sec",
            "range": "stddev: 0.0008669351783174135",
            "extra": "mean: 104.87975449999567 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.715657724787889,
            "unit": "iter/sec",
            "range": "stddev: 0.0007531965654429009",
            "extra": "mean: 582.8668419999872 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "75a26e27caf5f1d900e5245e53b1eddd8a8e2bd5",
          "message": "feat: implement create_cached_function (#397)\n\n* docs: add caching usage notebook\r\n* docs: only create tensorflow.inv is non-existent\r\n* feat: implement _collect_constant_sub_expressions\r\n* feat: implement extract_constant_sub_expressions\r\n* feat: implement prepare_caching\r\n* feat: implement create_cached_function\r\n* fix: switch bic and aic (same as in cell below)\r\n* fix: update compwa-org URL\r\n* style: define symbols in test_sympy globally\r\n* test: check cache transformer on domain variables only",
          "timestamp": "2022-01-07T16:38:18Z",
          "tree_id": "e52f45a5d1f0f73c54dc1495613fd5f1912bad38",
          "url": "https://github.com/ComPWA/tensorwaves/commit/75a26e27caf5f1d900e5245e53b1eddd8a8e2bd5"
        },
        "date": 1641573719848,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.23545399406688292,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.247114193000016 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2655441873108964,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.765851590000011 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20771061315035505,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.814390486999969 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.44651367015667065,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.239573089999965 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.484434422138822,
            "unit": "iter/sec",
            "range": "stddev: 0.00045984396374060954",
            "extra": "mean: 51.32301909999342 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 121.2935290613938,
            "unit": "iter/sec",
            "range": "stddev: 0.019958479807137358",
            "extra": "mean: 8.244462897058929 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.8393051695533824,
            "unit": "iter/sec",
            "range": "stddev: 0.12445904871914396",
            "extra": "mean: 260.46379639999486 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 89.86477168249391,
            "unit": "iter/sec",
            "range": "stddev: 0.00024116465674668764",
            "extra": "mean: 11.127831087505058 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.125692281897889,
            "unit": "iter/sec",
            "range": "stddev: 0.0019451634072701653",
            "extra": "mean: 123.0664373333165 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.178125900938014,
            "unit": "iter/sec",
            "range": "stddev: 0.0003194047273933318",
            "extra": "mean: 98.24991454545088 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.065410177846127,
            "unit": "iter/sec",
            "range": "stddev: 0.0013655932314320487",
            "extra": "mean: 99.35014890908178 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4472639314151454,
            "unit": "iter/sec",
            "range": "stddev: 0.0017517990109750018",
            "extra": "mean: 690.9589731999972 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.7031292049151014,
            "unit": "iter/sec",
            "range": "stddev: 0.002886250093873891",
            "extra": "mean: 149.18405559999428 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.22199125427404,
            "unit": "iter/sec",
            "range": "stddev: 0.0013448643742776318",
            "extra": "mean: 108.43645070000889 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.454930150554906,
            "unit": "iter/sec",
            "range": "stddev: 0.001237941141874551",
            "extra": "mean: 105.76492729999813 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6330836839285412,
            "unit": "iter/sec",
            "range": "stddev: 0.008894014277933597",
            "extra": "mean: 612.3384918000056 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "80a80fef7aa75480aad1d08bbe674a7339df9c1c",
          "message": "feat: add minuit_modifier constructor argument (#399)",
          "timestamp": "2022-01-11T16:14:48Z",
          "tree_id": "282c55e1aa07750fe62339eca3db80e5e2190cbd",
          "url": "https://github.com/ComPWA/tensorwaves/commit/80a80fef7aa75480aad1d08bbe674a7339df9c1c"
        },
        "date": 1641918004673,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.15118854449188074,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.614257735999985 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.17684321170598075,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.654726525000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.13880206170258688,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 7.20450393699997 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.31307326140480785,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.1941405520000217 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 12.232878332372955,
            "unit": "iter/sec",
            "range": "stddev: 0.0044857039622561885",
            "extra": "mean: 81.7469096666817 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 71.43154713560669,
            "unit": "iter/sec",
            "range": "stddev: 0.025926377172073155",
            "extra": "mean: 13.999416785717736 msec\nrounds: 84"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.564715842900643,
            "unit": "iter/sec",
            "range": "stddev: 0.1493812859650312",
            "extra": "mean: 389.9067425999988 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 51.93387872042472,
            "unit": "iter/sec",
            "range": "stddev: 0.0019035619309433524",
            "extra": "mean: 19.255253499999355 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.90513514733342,
            "unit": "iter/sec",
            "range": "stddev: 0.013904839558646563",
            "extra": "mean: 203.86798119999412 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.8421029919569705,
            "unit": "iter/sec",
            "range": "stddev: 0.00879922306968123",
            "extra": "mean: 206.52183599999034 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.690569843108173,
            "unit": "iter/sec",
            "range": "stddev: 0.006385925678012689",
            "extra": "mean: 213.19371280001178 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.7507277336187976,
            "unit": "iter/sec",
            "range": "stddev: 0.029219734231673084",
            "extra": "mean: 1.3320408387999918 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.499224295937978,
            "unit": "iter/sec",
            "range": "stddev: 0.00905447704305243",
            "extra": "mean: 222.26053519999596 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.86547222307756,
            "unit": "iter/sec",
            "range": "stddev: 0.010073474644411903",
            "extra": "mean: 205.52989599999592 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.6657256198772545,
            "unit": "iter/sec",
            "range": "stddev: 0.013581114833068886",
            "extra": "mean: 214.32893433332842 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.859439014267539,
            "unit": "iter/sec",
            "range": "stddev: 0.023166846784496015",
            "extra": "mean: 1.1635496915999965 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "730b89a2463f07dca67c07e61d14f9f19f67e2a2",
          "message": "ci: change upgrade cron job to bi-weekly (#398)\n\nSee https://crontab.guru/#0_2_*/14_*_*\r\n\r\n* ci: update pip constraints and pre-commit config\r\n* fix: enforce integer seed in NumpyUniformRNG\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-01-15T19:33:23+01:00",
          "tree_id": "fb2ceba89b4d20b07f2d05007e1f51061c234a91",
          "url": "https://github.com/ComPWA/tensorwaves/commit/730b89a2463f07dca67c07e61d14f9f19f67e2a2"
        },
        "date": 1642271828886,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.23108168956445013,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.327473984999983 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2501950511862246,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9968816139999603 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20068591648693698,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.982910696999966 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4074970983639368,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4540052039999978 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.04780748697151,
            "unit": "iter/sec",
            "range": "stddev: 0.0006440076554827827",
            "extra": "mean: 55.40839244444998 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.43887775301104,
            "unit": "iter/sec",
            "range": "stddev: 0.00012170914038414619",
            "extra": "mean: 7.223404409447041 msec\nrounds: 127"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.297510471865967,
            "unit": "iter/sec",
            "range": "stddev: 0.00041530891187143993",
            "extra": "mean: 232.6928594000151 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.93422875996384,
            "unit": "iter/sec",
            "range": "stddev: 0.00022239192426014602",
            "extra": "mean: 12.05774762666806 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.416914584329574,
            "unit": "iter/sec",
            "range": "stddev: 0.001055740153269013",
            "extra": "mean: 134.82695380000678 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.411585296707777,
            "unit": "iter/sec",
            "range": "stddev: 0.00055870356669898",
            "extra": "mean: 106.2520253999935 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.328857233198576,
            "unit": "iter/sec",
            "range": "stddev: 0.0004291207662271506",
            "extra": "mean: 107.19426560000329 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3257484376849527,
            "unit": "iter/sec",
            "range": "stddev: 0.0034133595281112157",
            "extra": "mean: 754.2909133999956 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.229290730752379,
            "unit": "iter/sec",
            "range": "stddev: 0.004859520990240447",
            "extra": "mean: 160.5319197999961 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.83221502067504,
            "unit": "iter/sec",
            "range": "stddev: 0.0005764421807414656",
            "extra": "mean: 113.22188122222263 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.642704967527338,
            "unit": "iter/sec",
            "range": "stddev: 0.001411497014031551",
            "extra": "mean: 115.70451655554986 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4106315899361626,
            "unit": "iter/sec",
            "range": "stddev: 0.02911826286996975",
            "extra": "mean: 708.9023151999982 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "94a7317a6ef458691ac54651f9a7f0530b98f424",
          "message": "docs: add notebook button for Deepnote (#400)\n\n* ci: build documentation on Python 3.8\r\n* ci: update pip constraints and pre-commit config\r\n* docs: add notebook button for Deepnote\r\n* docs: simplify install cells\r\n* fix: limit Sphinx on Python <=3.7\r\n  Causes conflicts with importlib-metadata\r\n  https://github.com/ComPWA/tensorwaves/actions/runs/1704199025\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-01-16T21:42:39+01:00",
          "tree_id": "e9d4ec872ecc0d99836b890e185cfaf937448a3a",
          "url": "https://github.com/ComPWA/tensorwaves/commit/94a7317a6ef458691ac54651f9a7f0530b98f424"
        },
        "date": 1642365981758,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22042903918910114,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.536607352999994 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2541937459963322,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9340070939999805 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20918333609188572,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.780495515000013 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41166555996144105,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.429156328000005 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.026727821275394,
            "unit": "iter/sec",
            "range": "stddev: 0.0010006697735116443",
            "extra": "mean: 52.55764466666809 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.80308188913594,
            "unit": "iter/sec",
            "range": "stddev: 0.00010985649401337777",
            "extra": "mean: 7.204450984731842 msec\nrounds: 131"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.6525684782277623,
            "unit": "iter/sec",
            "range": "stddev: 0.09329658156738484",
            "extra": "mean: 273.77994580000404 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 84.83526045699445,
            "unit": "iter/sec",
            "range": "stddev: 0.0001762224429807388",
            "extra": "mean: 11.787551480518294 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.455916561542078,
            "unit": "iter/sec",
            "range": "stddev: 0.0010817558628240895",
            "extra": "mean: 134.12167259999137 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.632746733670665,
            "unit": "iter/sec",
            "range": "stddev: 0.0003311760532767271",
            "extra": "mean: 103.81254980000278 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.484271124009284,
            "unit": "iter/sec",
            "range": "stddev: 0.0009163330835856124",
            "extra": "mean: 105.43772810000291 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.332768863781733,
            "unit": "iter/sec",
            "range": "stddev: 0.0022796429921870913",
            "extra": "mean: 750.3176485999973 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.234173918298012,
            "unit": "iter/sec",
            "range": "stddev: 0.00012915866030107807",
            "extra": "mean: 160.40617620000717 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.011575864834906,
            "unit": "iter/sec",
            "range": "stddev: 0.0002914309476822133",
            "extra": "mean: 110.96838277777958 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.992373983970907,
            "unit": "iter/sec",
            "range": "stddev: 0.0010352542891862377",
            "extra": "mean: 111.20533930000249 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5188786154420175,
            "unit": "iter/sec",
            "range": "stddev: 0.0012985425695055748",
            "extra": "mean: 658.380459 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "distinct": true,
          "id": "c5e7a6e47d0d41d326d571b887720d2749e4b2f4",
          "message": "fix: install optional dependencies jax and pwa\n\nFix-up to #400",
          "timestamp": "2022-01-16T21:44:45+01:00",
          "tree_id": "3a8c63dfeb7f0aa02b589134853ce20bae1fb5ca",
          "url": "https://github.com/ComPWA/tensorwaves/commit/c5e7a6e47d0d41d326d571b887720d2749e4b2f4"
        },
        "date": 1642366156012,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22013295177492187,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.542709266999992 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2530204038075029,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.95225043100001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2065465268689269,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.841524160000006 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4063383409393688,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4610033049999913 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.000144973167203,
            "unit": "iter/sec",
            "range": "stddev: 0.0007167277785009088",
            "extra": "mean: 55.55510811111238 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.51939420700293,
            "unit": "iter/sec",
            "range": "stddev: 0.00011811111757050976",
            "extra": "mean: 7.219205698413633 msec\nrounds: 126"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.2534155445738815,
            "unit": "iter/sec",
            "range": "stddev: 0.0007977584501536845",
            "extra": "mean: 235.1051736000045 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 83.42968404684356,
            "unit": "iter/sec",
            "range": "stddev: 0.00014000490798782433",
            "extra": "mean: 11.98614152054713 msec\nrounds: 73"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.562806636074707,
            "unit": "iter/sec",
            "range": "stddev: 0.0005607624184879005",
            "extra": "mean: 132.22604360000219 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.44985788793444,
            "unit": "iter/sec",
            "range": "stddev: 0.0012447986867738623",
            "extra": "mean: 105.82169719999683 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.472718713143813,
            "unit": "iter/sec",
            "range": "stddev: 0.0016084218755532149",
            "extra": "mean: 105.56631420000429 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3284834970837855,
            "unit": "iter/sec",
            "range": "stddev: 0.0033487498458724873",
            "extra": "mean: 752.7379920000101 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.2253636369991385,
            "unit": "iter/sec",
            "range": "stddev: 0.00038787551779707945",
            "extra": "mean: 160.63318679999838 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.907872669684775,
            "unit": "iter/sec",
            "range": "stddev: 0.0012751938510702791",
            "extra": "mean: 112.2602485555496 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.904183347392964,
            "unit": "iter/sec",
            "range": "stddev: 0.0003987511495013415",
            "extra": "mean: 112.3067619999972 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5169235705570614,
            "unit": "iter/sec",
            "range": "stddev: 0.0018633953839771416",
            "extra": "mean: 659.2289944000072 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "aa2e50a9fa80952186d7e009861f938971a1d8f2",
          "message": "docs: illustrate Hesse from FitResult.specifics (#401)\n\n* fix: avoid callback writing if calling hessian",
          "timestamp": "2022-01-18T11:25:06+01:00",
          "tree_id": "7367fbfd1725b76950205a124bc73b172e6cb1ad",
          "url": "https://github.com/ComPWA/tensorwaves/commit/aa2e50a9fa80952186d7e009861f938971a1d8f2"
        },
        "date": 1642501725120,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2286852303917936,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.372822845999963 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.261474182587914,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8244693610000127 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21305349158322806,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.693656942999951 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4465457727491069,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.2394120849999695 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.931301816665652,
            "unit": "iter/sec",
            "range": "stddev: 0.000832948837550801",
            "extra": "mean: 59.06220388887581 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 134.49908435480282,
            "unit": "iter/sec",
            "range": "stddev: 0.00017876504970193547",
            "extra": "mean: 7.434994853660437 msec\nrounds: 123"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.143787489923022,
            "unit": "iter/sec",
            "range": "stddev: 0.0007765301323185492",
            "extra": "mean: 241.32511679998743 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 81.06545777179961,
            "unit": "iter/sec",
            "range": "stddev: 0.00027586842998078",
            "extra": "mean: 12.335710270273362 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.28021368324071,
            "unit": "iter/sec",
            "range": "stddev: 0.0027137532232155742",
            "extra": "mean: 137.3586056000022 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.196358204263138,
            "unit": "iter/sec",
            "range": "stddev: 0.002875421773738228",
            "extra": "mean: 108.73869609998792 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.74356262415238,
            "unit": "iter/sec",
            "range": "stddev: 0.003432958786691366",
            "extra": "mean: 114.36985619999973 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3616677510535025,
            "unit": "iter/sec",
            "range": "stddev: 0.004683200991103128",
            "extra": "mean: 734.393539999985 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.5423812844518086,
            "unit": "iter/sec",
            "range": "stddev: 0.00034817862936391714",
            "extra": "mean: 180.42786099998693 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.044878489157115,
            "unit": "iter/sec",
            "range": "stddev: 0.0005716565482151104",
            "extra": "mean: 110.55980477778525 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.98349059913541,
            "unit": "iter/sec",
            "range": "stddev: 0.0007284285989475267",
            "extra": "mean: 111.31530544443852 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.55285560592762,
            "unit": "iter/sec",
            "range": "stddev: 0.00207419251175607",
            "extra": "mean: 643.9748784000017 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b1fecc499ea638384d0035c44f125ab065719a00",
          "message": "fix: use xreplace in prepare_caching() (#403)\n\n* docs: explain arguments of create_cached_function()\r\n* docs: link caching functions to caching notebook\r\n* docs: remove background in  expression tree visualization\r\n* fix: pass on use_cse to cache_converter",
          "timestamp": "2022-01-25T17:28:20Z",
          "tree_id": "c3412d20aa5b9eefb7bb7cedde84409ee3dcadae",
          "url": "https://github.com/ComPWA/tensorwaves/commit/b1fecc499ea638384d0035c44f125ab065719a00"
        },
        "date": 1643132024587,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.14558513621111968,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.868833082999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.17616740795008096,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.676418877000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.1380745851249739,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 7.2424624640000275 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.30380479009601835,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.29158733700001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 11.837551341581085,
            "unit": "iter/sec",
            "range": "stddev: 0.003856760108540739",
            "extra": "mean: 84.47693033333319 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 91.08108548281606,
            "unit": "iter/sec",
            "range": "stddev: 0.001750661018379039",
            "extra": "mean: 10.979227956046554 msec\nrounds: 91"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.9047472780314516,
            "unit": "iter/sec",
            "range": "stddev: 0.01811984957369812",
            "extra": "mean: 344.2640286000028 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 54.189490465239594,
            "unit": "iter/sec",
            "range": "stddev: 0.004037508733925805",
            "extra": "mean: 18.453762739132237 msec\nrounds: 46"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.990815736185657,
            "unit": "iter/sec",
            "range": "stddev: 0.011901106541588049",
            "extra": "mean: 200.3680465999878 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.518396677375035,
            "unit": "iter/sec",
            "range": "stddev: 0.015615262855539224",
            "extra": "mean: 221.31744319999598 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.733788474850229,
            "unit": "iter/sec",
            "range": "stddev: 0.01273183181991816",
            "extra": "mean: 211.2472928000102 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.7536225499892003,
            "unit": "iter/sec",
            "range": "stddev: 0.041899483526081034",
            "extra": "mean: 1.326924200999997 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.42950384609195,
            "unit": "iter/sec",
            "range": "stddev: 0.01297179211475556",
            "extra": "mean: 225.75891899998624 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.7597432265127555,
            "unit": "iter/sec",
            "range": "stddev: 0.009530154484745128",
            "extra": "mean: 210.09536700000808 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.777436462641022,
            "unit": "iter/sec",
            "range": "stddev: 0.0022163743372875674",
            "extra": "mean: 209.31727880001745 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.8929617249397945,
            "unit": "iter/sec",
            "range": "stddev: 0.036279236626549784",
            "extra": "mean: 1.1198688275999984 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0dbb55119ef9b6d0ea4b480e0ac2dcdd731a08b2",
          "message": "docs: abbreviate type aliases in API (#404)\n\n* build: block coverage v6.3\r\n  Freezes pytest: https://github.com/ComPWA/tensorwaves/runs/4988761243\r\n* ci: run RTD on Python 3.8\r\n* ci: update pip constraints and pre-commit config\r\n* docs: shorten type hint links\r\n* docs: sort API by location in the source code\r\n* docs: support type aliases in API\r\n* fix: fetch JAX object.inv from latest\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-01-29T11:12:06Z",
          "tree_id": "128c40cac7f64aeb7640b9d19ef47003564f58b1",
          "url": "https://github.com/ComPWA/tensorwaves/commit/0dbb55119ef9b6d0ea4b480e0ac2dcdd731a08b2"
        },
        "date": 1643455026312,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.14849409900844293,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.734274335999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.1827513751666792,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.4719150490000175 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.15300065204685684,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.535919858 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3499092815399209,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.8578836080000087 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 10.919036261138618,
            "unit": "iter/sec",
            "range": "stddev: 0.001833378145678251",
            "extra": "mean: 91.5831742000023 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 90.22086961421797,
            "unit": "iter/sec",
            "range": "stddev: 0.0004915544618767637",
            "extra": "mean: 11.083910011907149 msec\nrounds: 84"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.669513763604481,
            "unit": "iter/sec",
            "range": "stddev: 0.12301960482406957",
            "extra": "mean: 374.6000540000068 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 58.29375794394876,
            "unit": "iter/sec",
            "range": "stddev: 0.000908968396542382",
            "extra": "mean: 17.154495357144942 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.187758671634048,
            "unit": "iter/sec",
            "range": "stddev: 0.002184727897296314",
            "extra": "mean: 192.7614723999909 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.697387636928121,
            "unit": "iter/sec",
            "range": "stddev: 0.002595103337433158",
            "extra": "mean: 212.8842832000032 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.522642411139886,
            "unit": "iter/sec",
            "range": "stddev: 0.004072306924420519",
            "extra": "mean: 221.10967639998762 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8261159008572132,
            "unit": "iter/sec",
            "range": "stddev: 0.011655907900424112",
            "extra": "mean: 1.2104839029999994 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.482827621918283,
            "unit": "iter/sec",
            "range": "stddev: 0.003618768478325626",
            "extra": "mean: 223.07348939999656 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.751062213508088,
            "unit": "iter/sec",
            "range": "stddev: 0.0024296487418152203",
            "extra": "mean: 210.4792476000057 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.756177726002837,
            "unit": "iter/sec",
            "range": "stddev: 0.003294808491622978",
            "extra": "mean: 210.25286640001468 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9597886842621398,
            "unit": "iter/sec",
            "range": "stddev: 0.011146642526383326",
            "extra": "mean: 1.0418960093999998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0a5c933be247fe126cea11ed378ee61ddd5c235c",
          "message": "docs: automatically link code examples to API (#405)\n\n* build: install jupyterlab-myst\r\n* build: install sphinx-codeautolink\r\n* ci: update pip constraints and pre-commit config\r\n* ci: use black --preview flag\r\n* docs: activate sphinx_codeautolink extension\r\n* fix: skip code blocks with cell magic\r\n* style: format with black 22.1.0 style\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-01-30T23:31:20+01:00",
          "tree_id": "da8820b2cef629d38e7d4fd76d54f5c0dc11c5ed",
          "url": "https://github.com/ComPWA/tensorwaves/commit/0a5c933be247fe126cea11ed378ee61ddd5c235c"
        },
        "date": 1643582146005,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.18083695487862475,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.52984317100001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.21843780578002647,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.577962117999988 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.17978970754822754,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.562053655 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3438653402365587,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.9081151340000133 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 13.250658411967525,
            "unit": "iter/sec",
            "range": "stddev: 0.001424875553120078",
            "extra": "mean: 75.46794799999036 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 118.45173508691586,
            "unit": "iter/sec",
            "range": "stddev: 0.0001835634350775334",
            "extra": "mean: 8.44225708695811 msec\nrounds: 115"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.743004796589482,
            "unit": "iter/sec",
            "range": "stddev: 0.0018834845349332476",
            "extra": "mean: 267.1650330000034 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 72.68389091493533,
            "unit": "iter/sec",
            "range": "stddev: 0.00032344171600268574",
            "extra": "mean: 13.758206769232228 msec\nrounds: 65"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.163304660809561,
            "unit": "iter/sec",
            "range": "stddev: 0.0008282786109706674",
            "extra": "mean: 162.25061960001312 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 7.964908960767371,
            "unit": "iter/sec",
            "range": "stddev: 0.00046498016564875815",
            "extra": "mean: 125.55071312499422 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.05193786212225,
            "unit": "iter/sec",
            "range": "stddev: 0.0009022800891916596",
            "extra": "mean: 124.19370555555058 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.1333248786390777,
            "unit": "iter/sec",
            "range": "stddev: 0.007246934656795965",
            "extra": "mean: 882.359523599996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.396436446355697,
            "unit": "iter/sec",
            "range": "stddev: 0.0018324397597348516",
            "extra": "mean: 185.30747280000242 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.557380309803066,
            "unit": "iter/sec",
            "range": "stddev: 0.002142455024277927",
            "extra": "mean: 132.32098412499482 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.556177502007078,
            "unit": "iter/sec",
            "range": "stddev: 0.0018723624547801133",
            "extra": "mean: 132.34204724999898 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2940239416091477,
            "unit": "iter/sec",
            "range": "stddev: 0.005615272407146783",
            "extra": "mean: 772.783228999981 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a7acc756c76e8fc953e881d4a93841b1e93bd7e3",
          "message": "build: upgrade to AmpForm v0.12.3 (#406)\n\n* ci: update pip constraints and pre-commit config\r\n* fix: adapt to AmpForm v0.12.3\r\n* fix: update mass and helicit angle indices\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-01-31T14:12:12+01:00",
          "tree_id": "f6a3e37fca6ac86b90afeb5786cd38b0f2d0a7aa",
          "url": "https://github.com/ComPWA/tensorwaves/commit/a7acc756c76e8fc953e881d4a93841b1e93bd7e3"
        },
        "date": 1643635013040,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.16582403954815805,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.030488719999994 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.1634868914506081,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.116698355000011 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.15636368861753872,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.395346699999976 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3682586481643665,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.715482731999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 12.129132052387948,
            "unit": "iter/sec",
            "range": "stddev: 0.002184292155365285",
            "extra": "mean: 82.44613016667775 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 101.47692498803951,
            "unit": "iter/sec",
            "range": "stddev: 0.0006007283479196933",
            "extra": "mean: 9.854457061227112 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.025733906206626,
            "unit": "iter/sec",
            "range": "stddev: 0.11292798669170963",
            "extra": "mean: 330.4983290000223 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 61.63411635954786,
            "unit": "iter/sec",
            "range": "stddev: 0.0010850886787217742",
            "extra": "mean: 16.224780349999907 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.121135571253535,
            "unit": "iter/sec",
            "range": "stddev: 0.0013103366074531792",
            "extra": "mean: 195.26919099999986 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.667938292723767,
            "unit": "iter/sec",
            "range": "stddev: 0.0026250744773586815",
            "extra": "mean: 214.22733919999928 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.660993080917968,
            "unit": "iter/sec",
            "range": "stddev: 0.0026927030419027164",
            "extra": "mean: 214.5465532000003 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8380394879413121,
            "unit": "iter/sec",
            "range": "stddev: 0.018817247119406504",
            "extra": "mean: 1.1932611940000015 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.373396430784127,
            "unit": "iter/sec",
            "range": "stddev: 0.0026685428361917984",
            "extra": "mean: 228.65523760001452 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.667414049098646,
            "unit": "iter/sec",
            "range": "stddev: 0.0008829633550850528",
            "extra": "mean: 214.2514012000106 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.685575448204467,
            "unit": "iter/sec",
            "range": "stddev: 0.0029143228747621906",
            "extra": "mean: 213.42095780000818 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9823187003294137,
            "unit": "iter/sec",
            "range": "stddev: 0.014340874356731972",
            "extra": "mean: 1.017999555200015 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b73659a51539f3663820637ade3b85bdd05c097e",
          "message": "docs: show second level in left sidebar (#407)\n\n* fix: link to graphviz API in code examples",
          "timestamp": "2022-01-31T18:36:28Z",
          "tree_id": "cf429458e4e121f28f9eccf8d37baf9c81e72fee",
          "url": "https://github.com/ComPWA/tensorwaves/commit/b73659a51539f3663820637ade3b85bdd05c097e"
        },
        "date": 1643654421508,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21898506417253302,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.566521482999974 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.1997941108501278,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.005152533 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20542347704167155,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.867992764999997 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41071789203740655,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4347612299999923 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.90976677135204,
            "unit": "iter/sec",
            "range": "stddev: 0.0007491131978742326",
            "extra": "mean: 62.854472624994884 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 140.07257442386174,
            "unit": "iter/sec",
            "range": "stddev: 0.00011667626481534113",
            "extra": "mean: 7.139156284612752 msec\nrounds: 130"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.642053127267159,
            "unit": "iter/sec",
            "range": "stddev: 0.10149338754516611",
            "extra": "mean: 274.5704043999922 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 84.90116548370112,
            "unit": "iter/sec",
            "range": "stddev: 0.0001696196362861292",
            "extra": "mean: 11.778401324678807 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.606191573200918,
            "unit": "iter/sec",
            "range": "stddev: 0.00023739867086483728",
            "extra": "mean: 131.4718398000025 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.398261504887572,
            "unit": "iter/sec",
            "range": "stddev: 0.00043273395754134746",
            "extra": "mean: 106.4026575000014 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.3956848379992,
            "unit": "iter/sec",
            "range": "stddev: 0.0008775658085712443",
            "extra": "mean: 106.43183729999919 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3244727697933287,
            "unit": "iter/sec",
            "range": "stddev: 0.0010382505078226462",
            "extra": "mean: 755.017409800007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.255682599516615,
            "unit": "iter/sec",
            "range": "stddev: 0.00017850057042828165",
            "extra": "mean: 159.85465759999897 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.905882270137043,
            "unit": "iter/sec",
            "range": "stddev: 0.0007184889884492926",
            "extra": "mean: 112.28533790000483 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.87438186191995,
            "unit": "iter/sec",
            "range": "stddev: 0.001219451802003387",
            "extra": "mean: 112.68390470000043 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5110922492681196,
            "unit": "iter/sec",
            "range": "stddev: 0.0035047893231384365",
            "extra": "mean: 661.7729662000045 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1df3dad9f0c870b13d93a6d89220fe15292634f7",
          "message": "fix: run upgrade cron job on even weeks only (#408)\n\n* ci: update pip constraints and pre-commit config\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-02-01T13:25:25Z",
          "tree_id": "74a95c2bc62ef91a9321d88df071965106b40067",
          "url": "https://github.com/ComPWA/tensorwaves/commit/1df3dad9f0c870b13d93a6d89220fe15292634f7"
        },
        "date": 1643722221208,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2271502966700782,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.402371533999968 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.20022776805636747,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.9943122760000165 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21078810809639043,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.744100647000039 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.39159280003710084,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5536731009999585 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.625191627934583,
            "unit": "iter/sec",
            "range": "stddev: 0.000397613948428707",
            "extra": "mean: 60.14968262499565 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.7740337719343,
            "unit": "iter/sec",
            "range": "stddev: 0.00010488419863410316",
            "extra": "mean: 7.258261753846595 msec\nrounds: 130"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.7751340554256956,
            "unit": "iter/sec",
            "range": "stddev: 0.0855540647374634",
            "extra": "mean: 264.8912555999914 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 86.78272245252347,
            "unit": "iter/sec",
            "range": "stddev: 0.00013242965918640199",
            "extra": "mean: 11.523030987499538 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.257942693252586,
            "unit": "iter/sec",
            "range": "stddev: 0.00024281480557370412",
            "extra": "mean: 121.09553640001423 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.133175059771467,
            "unit": "iter/sec",
            "range": "stddev: 0.0002544183782787655",
            "extra": "mean: 98.68575190909144 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.087896511775828,
            "unit": "iter/sec",
            "range": "stddev: 0.0007004339032546955",
            "extra": "mean: 99.12869336364399 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.5008289566097126,
            "unit": "iter/sec",
            "range": "stddev: 0.0019564229692426364",
            "extra": "mean: 666.298444999984 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.460442669479705,
            "unit": "iter/sec",
            "range": "stddev: 0.00034606694568234685",
            "extra": "mean: 154.78815480000776 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.274306813476235,
            "unit": "iter/sec",
            "range": "stddev: 0.0003873180565294235",
            "extra": "mean: 107.82477010000662 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.27271228195543,
            "unit": "iter/sec",
            "range": "stddev: 0.000743394797890273",
            "extra": "mean: 107.84331159999283 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.7015508022832795,
            "unit": "iter/sec",
            "range": "stddev: 0.0015184992413123707",
            "extra": "mean: 587.6991734000057 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "90e582a0995f611ef439fee7f3be155c01a08f79",
          "message": "docs: add Hypothesis and utterances overlay (#409)",
          "timestamp": "2022-02-03T09:39:59Z",
          "tree_id": "c752b1b3cbaae7f004d58eb0eccebc4f48ebef2e",
          "url": "https://github.com/ComPWA/tensorwaves/commit/90e582a0995f611ef439fee7f3be155c01a08f79"
        },
        "date": 1643881443147,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.20431244842890772,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.894464373999995 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19544890348435406,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.1164267599999675 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.19307717081379583,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.179276222999988 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4063469471943433,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.460951181999974 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.78798893810575,
            "unit": "iter/sec",
            "range": "stddev: 0.0010338094900417011",
            "extra": "mean: 63.33928937500133 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 140.11740959418523,
            "unit": "iter/sec",
            "range": "stddev: 0.00011867344740512862",
            "extra": "mean: 7.136871876922704 msec\nrounds: 130"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.202830729219067,
            "unit": "iter/sec",
            "range": "stddev: 0.0031000891510240326",
            "extra": "mean: 237.93487399998412 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 83.47274063580073,
            "unit": "iter/sec",
            "range": "stddev: 0.00027087503824623665",
            "extra": "mean: 11.9799588749948 msec\nrounds: 72"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.245270206824045,
            "unit": "iter/sec",
            "range": "stddev: 0.0005051178423991455",
            "extra": "mean: 138.02107740000338 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.49169287134725,
            "unit": "iter/sec",
            "range": "stddev: 0.0006796241998093145",
            "extra": "mean: 105.35528420001015 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.071971490624636,
            "unit": "iter/sec",
            "range": "stddev: 0.0012298955159408499",
            "extra": "mean: 110.22962330001178 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2719062822901792,
            "unit": "iter/sec",
            "range": "stddev: 0.016747790886250367",
            "extra": "mean: 786.221448799995 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.104270395083287,
            "unit": "iter/sec",
            "range": "stddev: 0.0004331835588199552",
            "extra": "mean: 163.8197418000118 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.05979968253617,
            "unit": "iter/sec",
            "range": "stddev: 0.0007929064056425009",
            "extra": "mean: 110.37771640002347 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.820463237145528,
            "unit": "iter/sec",
            "range": "stddev: 0.0015924742524820866",
            "extra": "mean: 113.37273033333555 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4617455265402681,
            "unit": "iter/sec",
            "range": "stddev: 0.015408792953043884",
            "extra": "mean: 684.1136038000059 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "921f7d4342c7d012a55f4c8195474de2507245d1",
          "message": "ci: manually run tests with latest AmpForm version (#410)\n\n* ci: include doctests in test coverage\r\n* ci: test tensorwaves with latest version of AmpForm\r\n* ci: test notebooks with nbmake on GitHub Actions\r\n* docs: unfold right sidebar unto second level\r\n* fix: remove tf from scipy URL\r\n* fix: remap scipy 1.7.3 to 1.7.1\r\n  https://github.com/ComPWA/tensorwaves/runs/5085332721",
          "timestamp": "2022-02-06T20:54:45+01:00",
          "tree_id": "4fbe23eacf3e193dd1b730cffde620229b767dd6",
          "url": "https://github.com/ComPWA/tensorwaves/commit/921f7d4342c7d012a55f4c8195474de2507245d1"
        },
        "date": 1644177546103,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.17654672025938448,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.664223036999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.16673333512966257,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.997600895000005 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.17534937282383012,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.702900352 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3425538845618679,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.919248752000044 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 12.985389307425127,
            "unit": "iter/sec",
            "range": "stddev: 0.0008205966500703413",
            "extra": "mean: 77.00962800000103 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 117.245108745514,
            "unit": "iter/sec",
            "range": "stddev: 0.00016040566823448968",
            "extra": "mean: 8.529140453701542 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.0579172910870884,
            "unit": "iter/sec",
            "range": "stddev: 0.11352699871094794",
            "extra": "mean: 327.01996320001854 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 70.28239059318351,
            "unit": "iter/sec",
            "range": "stddev: 0.00023138766462069035",
            "extra": "mean: 14.228315109375167 msec\nrounds: 64"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.411168119460812,
            "unit": "iter/sec",
            "range": "stddev: 0.00030167435088828277",
            "extra": "mean: 155.97781579998582 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 7.80399421027113,
            "unit": "iter/sec",
            "range": "stddev: 0.002395895028764366",
            "extra": "mean: 128.13951075000318 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 7.766110066775615,
            "unit": "iter/sec",
            "range": "stddev: 0.0015590886800032032",
            "extra": "mean: 128.7645927499952 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.1277572550379595,
            "unit": "iter/sec",
            "range": "stddev: 0.0020038389110363306",
            "extra": "mean: 886.715643399998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.360568210452846,
            "unit": "iter/sec",
            "range": "stddev: 0.00017498537581062582",
            "extra": "mean: 186.54738839999254 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.448073502535626,
            "unit": "iter/sec",
            "range": "stddev: 0.0019076229085390219",
            "extra": "mean: 134.2629069999859 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.35742726631112,
            "unit": "iter/sec",
            "range": "stddev: 0.0006612928154744099",
            "extra": "mean: 135.91707587499968 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2700976873231675,
            "unit": "iter/sec",
            "range": "stddev: 0.0031513478898541943",
            "extra": "mean: 787.3410132000004 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "distinct": true,
          "id": "3798166d246c627b83840bf096f27bbf0c3a0642",
          "message": "fix: add inputs layer to workflow_dispatch\n\nFix-up to https://github.com/ComPWA/tensorwaves/pull/410",
          "timestamp": "2022-02-06T20:56:19+01:00",
          "tree_id": "5afc5c1747ecb26c00f72f84ae44f140578838aa",
          "url": "https://github.com/ComPWA/tensorwaves/commit/3798166d246c627b83840bf096f27bbf0c3a0642"
        },
        "date": 1644177624622,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21342216280183082,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.685548992999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19560749310063685,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.112278595000021 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20258224792773316,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.936266677999981 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41530348696017577,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.407877687999985 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.428802019728984,
            "unit": "iter/sec",
            "range": "stddev: 0.0007815177872481923",
            "extra": "mean: 64.81384612501273 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 139.15702050145867,
            "unit": "iter/sec",
            "range": "stddev: 0.000136851544863415",
            "extra": "mean: 7.18612684000027 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.4988637565908878,
            "unit": "iter/sec",
            "range": "stddev: 0.12092244643937794",
            "extra": "mean: 285.8070704000056 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 83.88308246288366,
            "unit": "iter/sec",
            "range": "stddev: 0.00025403657665742284",
            "extra": "mean: 11.921354945944875 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.323101766012691,
            "unit": "iter/sec",
            "range": "stddev: 0.0006627348495213516",
            "extra": "mean: 136.55415859999493 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.599272531802354,
            "unit": "iter/sec",
            "range": "stddev: 0.0008037320890008378",
            "extra": "mean: 104.17456079999852 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.61808333859188,
            "unit": "iter/sec",
            "range": "stddev: 0.0009027065273807224",
            "extra": "mean: 103.97081879999632 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3256918834061964,
            "unit": "iter/sec",
            "range": "stddev: 0.0019652025055895586",
            "extra": "mean: 754.3230916000084 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.2538119407129775,
            "unit": "iter/sec",
            "range": "stddev: 0.00018469896283882957",
            "extra": "mean: 159.90247380000255 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.101815631093023,
            "unit": "iter/sec",
            "range": "stddev: 0.0015595060493928169",
            "extra": "mean: 109.86818899999093 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.023227029774523,
            "unit": "iter/sec",
            "range": "stddev: 0.0009000079560358332",
            "extra": "mean: 110.82509580000988 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5317691823462498,
            "unit": "iter/sec",
            "range": "stddev: 0.0009621949805946849",
            "extra": "mean: 652.8398739999943 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "distinct": true,
          "id": "c2fce84cdf23e011d4093276393c4c67a36e7163",
          "message": "fix: install graphviz in ptyest-notebook job\n\nFix-up to https://github.com/ComPWA/tensorwaves/pull/410",
          "timestamp": "2022-02-06T21:06:35+01:00",
          "tree_id": "d943809caae205aef654444a216a214d1977a862",
          "url": "https://github.com/ComPWA/tensorwaves/commit/c2fce84cdf23e011d4093276393c4c67a36e7163"
        },
        "date": 1644178309881,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.1975697202392894,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.061504357999979 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.18675154478062528,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.35470804900001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.1877732641898318,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.325571797000009 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41318266891197336,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.420237041000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.73612565034418,
            "unit": "iter/sec",
            "range": "stddev: 0.0005809691249398896",
            "extra": "mean: 67.86044200000723 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.2118227720897,
            "unit": "iter/sec",
            "range": "stddev: 0.00020713012083582917",
            "extra": "mean: 7.34150663025195 msec\nrounds: 119"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.264130959232548,
            "unit": "iter/sec",
            "range": "stddev: 0.001372561383667918",
            "extra": "mean: 234.51437340000894 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 84.94266496977755,
            "unit": "iter/sec",
            "range": "stddev: 0.0001729394038250292",
            "extra": "mean: 11.772646883115783 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.65127627542407,
            "unit": "iter/sec",
            "range": "stddev: 0.00010950467581877949",
            "extra": "mean: 130.697149599996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.63074149017385,
            "unit": "iter/sec",
            "range": "stddev: 0.0006207377498854002",
            "extra": "mean: 103.83416490000172 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.693839731197425,
            "unit": "iter/sec",
            "range": "stddev: 0.0007510583534790368",
            "extra": "mean: 103.1582971999967 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3625099814654305,
            "unit": "iter/sec",
            "range": "stddev: 0.0008989292447109533",
            "extra": "mean: 733.9395773999854 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.457758964913504,
            "unit": "iter/sec",
            "range": "stddev: 0.0005543453135769784",
            "extra": "mean: 154.8524814000075 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.131195243634954,
            "unit": "iter/sec",
            "range": "stddev: 0.000680817304328379",
            "extra": "mean: 109.51468819999945 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.169182837144549,
            "unit": "iter/sec",
            "range": "stddev: 0.0002772395518670126",
            "extra": "mean: 109.06097279999472 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5805201628349554,
            "unit": "iter/sec",
            "range": "stddev: 0.0007430790382745041",
            "extra": "mean: 632.7030958000023 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "77fbf2053a379801c1d37f49bf676c7faf51045c",
          "message": "docs: clarify pinned dependencies with Conda (#411)",
          "timestamp": "2022-02-08T16:39:05+01:00",
          "tree_id": "0a304bc00c0ea8a1e0ac009ef21d772c7eaa4b8f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/77fbf2053a379801c1d37f49bf676c7faf51045c"
        },
        "date": 1644335011369,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.17757500350802394,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.631423231000042 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.16917106790984424,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.9111762569999655 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.17899736135908606,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.586674531999961 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3456205356551868,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.8933465949999686 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 13.530881222387702,
            "unit": "iter/sec",
            "range": "stddev: 0.0036978499403609337",
            "extra": "mean: 73.90501649999237 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 120.43521077678584,
            "unit": "iter/sec",
            "range": "stddev: 0.0005821685403915138",
            "extra": "mean: 8.303219577980366 msec\nrounds: 109"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.0819024165711415,
            "unit": "iter/sec",
            "range": "stddev: 0.11077950427549178",
            "extra": "mean: 324.47490700000117 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 71.01008890980698,
            "unit": "iter/sec",
            "range": "stddev: 0.0011659074685891127",
            "extra": "mean: 14.082505955881054 msec\nrounds: 68"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.220725050188888,
            "unit": "iter/sec",
            "range": "stddev: 0.00256839118921056",
            "extra": "mean: 160.75296559998833 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.011750546102654,
            "unit": "iter/sec",
            "range": "stddev: 0.0016205083117573598",
            "extra": "mean: 124.8166670000046 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 7.941738804091651,
            "unit": "iter/sec",
            "range": "stddev: 0.0034420227862909146",
            "extra": "mean: 125.91700944442941 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.158586942245223,
            "unit": "iter/sec",
            "range": "stddev: 0.007899808675795443",
            "extra": "mean: 863.1203784000036 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.568123701649826,
            "unit": "iter/sec",
            "range": "stddev: 0.0010527129980346585",
            "extra": "mean: 179.5937112000047 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.650265598860747,
            "unit": "iter/sec",
            "range": "stddev: 0.003255992978492455",
            "extra": "mean: 130.71441600000355 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.791086105156663,
            "unit": "iter/sec",
            "range": "stddev: 0.00220512548137201",
            "extra": "mean: 128.35180955555518 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.3234505696103431,
            "unit": "iter/sec",
            "range": "stddev: 0.01171981384164036",
            "extra": "mean: 755.6005664000168 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e422bce8cf3b53ba35b08b6ec7e0f7bba505ed8c",
          "message": "docs: use create_cached_function() in PWA notebook (#412)\n\n* docs: assert if initial parameter names are correct\r\n* docs: extract safe_downcast_to_real() function\r\n* docs: call doit() only once\r\n* docs: make intensity array equality assert statement visible\r\n* docs: illustrate create_cached_function() in amplitude-analysis notebook",
          "timestamp": "2022-02-10T15:05:50+01:00",
          "tree_id": "b71122db59b3f466e995adc87c2065bfdebcd76d",
          "url": "https://github.com/ComPWA/tensorwaves/commit/e422bce8cf3b53ba35b08b6ec7e0f7bba505ed8c"
        },
        "date": 1644502171853,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22623276261657688,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.420226267999993 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.21691011481577768,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.610204557999992 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.22138440538675885,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.517029997000009 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.45833174538571353,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.181825740999983 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.033800220017145,
            "unit": "iter/sec",
            "range": "stddev: 0.005795161733549686",
            "extra": "mean: 62.36824622222533 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 154.73390320783622,
            "unit": "iter/sec",
            "range": "stddev: 0.0001328217661665524",
            "extra": "mean: 6.46270777941157 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.879024701474019,
            "unit": "iter/sec",
            "range": "stddev: 0.11355909898934888",
            "extra": "mean: 257.7967600000079 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 93.56351416329679,
            "unit": "iter/sec",
            "range": "stddev: 0.00017078387838568567",
            "extra": "mean: 10.687926901235196 msec\nrounds: 81"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.137237068009043,
            "unit": "iter/sec",
            "range": "stddev: 0.00043543803619965847",
            "extra": "mean: 122.89183560000083 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.184953996159534,
            "unit": "iter/sec",
            "range": "stddev: 0.0004580268832493049",
            "extra": "mean: 98.18404681818618 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.932811868957899,
            "unit": "iter/sec",
            "range": "stddev: 0.0008163529414265743",
            "extra": "mean: 100.6764260909046 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4338950671746933,
            "unit": "iter/sec",
            "range": "stddev: 0.0014173105421602598",
            "extra": "mean: 697.4011020000034 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.813383356685523,
            "unit": "iter/sec",
            "range": "stddev: 0.0002877504116795024",
            "extra": "mean: 146.76995959999317 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.643418803465202,
            "unit": "iter/sec",
            "range": "stddev: 0.0008230683995830205",
            "extra": "mean: 103.69766370000093 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.468561262399318,
            "unit": "iter/sec",
            "range": "stddev: 0.000987046917703232",
            "extra": "mean: 105.61266619999685 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.645208078947064,
            "unit": "iter/sec",
            "range": "stddev: 0.0012835858376006926",
            "extra": "mean: 607.8258506000054 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bea8cd9b06f4e833347716dbc63f8f55238ce46e",
          "message": "docs: show how to install from git with opt. deps (#413)",
          "timestamp": "2022-02-16T12:40:07+01:00",
          "tree_id": "57836584170781df16d6cf3ca017a6df0188962f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/bea8cd9b06f4e833347716dbc63f8f55238ce46e"
        },
        "date": 1645011836318,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22878293177676115,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.370955438999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.20052928626261426,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.986802769000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21596509093448996,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.63037797299998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.39115505758001434,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5565309220000074 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.686477548081513,
            "unit": "iter/sec",
            "range": "stddev: 0.0003824322239632149",
            "extra": "mean: 59.92876550000048 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.81089193019298,
            "unit": "iter/sec",
            "range": "stddev: 0.0001256102544074912",
            "extra": "mean: 7.309359553844912 msec\nrounds: 130"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.771469091575256,
            "unit": "iter/sec",
            "range": "stddev: 0.08526668994956452",
            "extra": "mean: 265.14866640000037 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 86.58742983136815,
            "unit": "iter/sec",
            "range": "stddev: 0.00011282562692328319",
            "extra": "mean: 11.549020474998883 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.216369276960394,
            "unit": "iter/sec",
            "range": "stddev: 0.0003365934294705833",
            "extra": "mean: 121.7082590000075 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.222197851676608,
            "unit": "iter/sec",
            "range": "stddev: 0.0003067110967634655",
            "extra": "mean: 97.82632018181724 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.145833124690457,
            "unit": "iter/sec",
            "range": "stddev: 0.0009463008219650751",
            "extra": "mean: 98.56263036363605 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4703978909897237,
            "unit": "iter/sec",
            "range": "stddev: 0.0025464599571250355",
            "extra": "mean: 680.0880265999979 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.524577463469962,
            "unit": "iter/sec",
            "range": "stddev: 0.00019930178060774512",
            "extra": "mean: 153.26663000000167 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.478852760535599,
            "unit": "iter/sec",
            "range": "stddev: 0.0002685411633040021",
            "extra": "mean: 105.4979990999982 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.257688195284542,
            "unit": "iter/sec",
            "range": "stddev: 0.0008316895712135506",
            "extra": "mean: 108.01832800000284 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6978728168833293,
            "unit": "iter/sec",
            "range": "stddev: 0.0022662532781697006",
            "extra": "mean: 588.9722657999982 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "distinct": true,
          "id": "b5aa344eb828d6be0e2ac83237d12af519e26ccc",
          "message": "ci: run pre-commit pylint hook serial",
          "timestamp": "2022-02-19T12:41:58+01:00",
          "tree_id": "afa7bd0fcbc6efe6761272596055e733f0cef1c8",
          "url": "https://github.com/ComPWA/tensorwaves/commit/b5aa344eb828d6be0e2ac83237d12af519e26ccc"
        },
        "date": 1645271263352,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.17508443575601482,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.711529957999971 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.16308317509875747,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 6.131840389999979 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.17364435366455871,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.758897302999969 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3344745239513334,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.989764326999989 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 13.31475688885927,
            "unit": "iter/sec",
            "range": "stddev: 0.0012590270206758755",
            "extra": "mean: 75.10463828571444 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 115.69713134004222,
            "unit": "iter/sec",
            "range": "stddev: 0.0001293571802099344",
            "extra": "mean: 8.643256651376497 msec\nrounds: 109"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.104583455883094,
            "unit": "iter/sec",
            "range": "stddev: 0.10460591304156604",
            "extra": "mean: 322.1044028000051 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 71.22037174189734,
            "unit": "iter/sec",
            "range": "stddev: 0.00018489957073802664",
            "extra": "mean: 14.040926430769 msec\nrounds: 65"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.26283361075036,
            "unit": "iter/sec",
            "range": "stddev: 0.0005594719393824531",
            "extra": "mean: 159.67213279999442 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 7.687626533888539,
            "unit": "iter/sec",
            "range": "stddev: 0.0004404027594267708",
            "extra": "mean: 130.07915974999662 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 7.603597463836495,
            "unit": "iter/sec",
            "range": "stddev: 0.00143880662629219",
            "extra": "mean: 131.51669387498544 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.1219328684229446,
            "unit": "iter/sec",
            "range": "stddev: 0.0014103404132566578",
            "extra": "mean: 891.3189266000018 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.290926761334517,
            "unit": "iter/sec",
            "range": "stddev: 0.00043435189185894106",
            "extra": "mean: 189.0028052000048 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.309006536643609,
            "unit": "iter/sec",
            "range": "stddev: 0.0004895576451399803",
            "extra": "mean: 136.817499750002 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.265611563939417,
            "unit": "iter/sec",
            "range": "stddev: 0.0011588877859058516",
            "extra": "mean: 137.63466312501293 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2918305149027431,
            "unit": "iter/sec",
            "range": "stddev: 0.00063789517669656",
            "extra": "mean: 774.0953541999943 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bea8cd9b06f4e833347716dbc63f8f55238ce46e",
          "message": "docs: show how to install from git with opt. deps (#413)",
          "timestamp": "2022-02-16T12:40:07+01:00",
          "tree_id": "57836584170781df16d6cf3ca017a6df0188962f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/bea8cd9b06f4e833347716dbc63f8f55238ce46e"
        },
        "date": 1645271410125,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21600919025270254,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.629432658999974 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19604492933145418,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.100871537000046 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20455648833173326,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.888625181999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41827175769772706,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.390790154000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.20228136749623,
            "unit": "iter/sec",
            "range": "stddev: 0.0005666549510459803",
            "extra": "mean: 61.71970337499033 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 140.5279360749191,
            "unit": "iter/sec",
            "range": "stddev: 0.000104742606714178",
            "extra": "mean: 7.116022820308654 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5880903379272873,
            "unit": "iter/sec",
            "range": "stddev: 0.11472161133442042",
            "extra": "mean: 278.699783400009 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 85.87087726349837,
            "unit": "iter/sec",
            "range": "stddev: 0.00013591947691033894",
            "extra": "mean: 11.64539168420812 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.618113037647277,
            "unit": "iter/sec",
            "range": "stddev: 0.00037010002395890436",
            "extra": "mean: 131.2661015999879 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.664279961912374,
            "unit": "iter/sec",
            "range": "stddev: 0.0006521822922598211",
            "extra": "mean: 103.47382360000665 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.540920683466233,
            "unit": "iter/sec",
            "range": "stddev: 0.0009825621567304311",
            "extra": "mean: 104.81168779999734 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.336575403341381,
            "unit": "iter/sec",
            "range": "stddev: 0.0013308850802185974",
            "extra": "mean: 748.1807591999996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.288417577972329,
            "unit": "iter/sec",
            "range": "stddev: 0.00019366010603997602",
            "extra": "mean: 159.0225184000019 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.174976329571074,
            "unit": "iter/sec",
            "range": "stddev: 0.0014593230154670612",
            "extra": "mean: 108.992106799991 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.118958318629943,
            "unit": "iter/sec",
            "range": "stddev: 0.0008414564558852222",
            "extra": "mean: 109.6616483000048 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.543055376218668,
            "unit": "iter/sec",
            "range": "stddev: 0.00078583273645864",
            "extra": "mean: 648.0648817999963 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0dd2d22ed5301387f4602a13737bd61537f78e5f",
          "message": "chore: switch to new import attrs API (#414)\n\n* chore: simplify _new_type_to_xref in Sphinx <4.4\r\n* ci: run pre-commit pylint hook serial\r\n* fix: run update cron job on Mondays",
          "timestamp": "2022-02-19T12:50:03+01:00",
          "tree_id": "cae961c762b5b51d681fef19633e62eac038f821",
          "url": "https://github.com/ComPWA/tensorwaves/commit/0dd2d22ed5301387f4602a13737bd61537f78e5f"
        },
        "date": 1645271639050,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.20116547272534632,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.971031988999982 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19005486765054802,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.2616384539999785 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.19705126675358686,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.074821473999975 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41134781227886663,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4310327420000135 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.352385910940045,
            "unit": "iter/sec",
            "range": "stddev: 0.0017937544535418257",
            "extra": "mean: 69.67482662501112 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.68813734783072,
            "unit": "iter/sec",
            "range": "stddev: 0.00013341177634887947",
            "extra": "mean: 7.21042202399758 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.45162695622447,
            "unit": "iter/sec",
            "range": "stddev: 0.12180652299136101",
            "extra": "mean: 289.7184465999885 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.71163385238445,
            "unit": "iter/sec",
            "range": "stddev: 0.0003313841672733407",
            "extra": "mean: 12.090197635131972 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.51851281530941,
            "unit": "iter/sec",
            "range": "stddev: 0.0002026194180180416",
            "extra": "mean: 133.00502699999015 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.333586122934983,
            "unit": "iter/sec",
            "range": "stddev: 0.0008878989780145615",
            "extra": "mean: 107.13995530000489 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.206764360399383,
            "unit": "iter/sec",
            "range": "stddev: 0.001100669371007413",
            "extra": "mean: 108.61579169998663 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3263196130381565,
            "unit": "iter/sec",
            "range": "stddev: 0.003465466628804476",
            "extra": "mean: 753.9660803999823 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.214855118601307,
            "unit": "iter/sec",
            "range": "stddev: 0.00023741737456570234",
            "extra": "mean: 160.9047968000027 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.839013666096626,
            "unit": "iter/sec",
            "range": "stddev: 0.0012167816464164901",
            "extra": "mean: 113.13479510000661 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.817092882095086,
            "unit": "iter/sec",
            "range": "stddev: 0.001516172023721838",
            "extra": "mean: 113.41606733333896 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5295027425160093,
            "unit": "iter/sec",
            "range": "stddev: 0.002158698933590611",
            "extra": "mean: 653.8072618000115 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "42db2fccef556f566b8a3a80c10c1c54e7ed199d",
          "message": "ci: update pip constraints and pre-commit config (#415)\n\n* ci: ignore SciPy deprecation warning\r\n  https://github.com/ComPWA/tensorwaves/runs/5271880587?check_suite_focus=true#step:6:357\r\n* docs: import IPython.display.display() in notebooks\r\n* fix: improve intersphinx fallback for scipy API\r\n* fix: remove leading zero from week number in cron job\r\n\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-02-21T11:35:58+01:00",
          "tree_id": "f195fc99b9707723ce9374c118dbf117580ac62d",
          "url": "https://github.com/ComPWA/tensorwaves/commit/42db2fccef556f566b8a3a80c10c1c54e7ed199d"
        },
        "date": 1645439984827,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.24086346190566113,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.151729748000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2105912922918226,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.748534419999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2148439849688246,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.654540363999985 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4876222975129924,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0507675819999918 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.88628191513547,
            "unit": "iter/sec",
            "range": "stddev: 0.0013610821773757692",
            "extra": "mean: 52.94848422222268 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 153.51126987920284,
            "unit": "iter/sec",
            "range": "stddev: 0.00013541572907298496",
            "extra": "mean: 6.514179713234699 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.751659724883148,
            "unit": "iter/sec",
            "range": "stddev: 0.12468441174801465",
            "extra": "mean: 266.5486939999994 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 93.67681970505305,
            "unit": "iter/sec",
            "range": "stddev: 0.00026540237223902647",
            "extra": "mean: 10.674999462498391 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.232433755310165,
            "unit": "iter/sec",
            "range": "stddev: 0.000495444720988576",
            "extra": "mean: 121.47076183333638 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.146666896853962,
            "unit": "iter/sec",
            "range": "stddev: 0.0010683545026448369",
            "extra": "mean: 98.55453127273316 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.118976376087737,
            "unit": "iter/sec",
            "range": "stddev: 0.0007750899502637338",
            "extra": "mean: 98.82422518181886 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4228750556117256,
            "unit": "iter/sec",
            "range": "stddev: 0.0029962277789358257",
            "extra": "mean: 702.8023971999971 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.46027855626698,
            "unit": "iter/sec",
            "range": "stddev: 0.0002569595727606086",
            "extra": "mean: 134.04325220000715 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.468601794988505,
            "unit": "iter/sec",
            "range": "stddev: 0.0008441289729167708",
            "extra": "mean: 105.61221410000314 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.439415993602983,
            "unit": "iter/sec",
            "range": "stddev: 0.0009408906908596512",
            "extra": "mean: 105.93875729999525 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6266033163753277,
            "unit": "iter/sec",
            "range": "stddev: 0.001164158559138677",
            "extra": "mean: 614.778040799996 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "542ace413be3f05ccb7d379822b9dc4c856da9b8",
          "message": "ci: run tests with specific dependency versions (#416)\n\n* ci: recommend GitHub Actions for VS Code\r\n* ci: add settings for GitHub Actions extension\r\n* ci: run core GitHub actions on version branches\r\n* ci: remove outdated VSCode settings\r\n* ci: run pytest in VSCode non-verbose",
          "timestamp": "2022-02-23T20:43:26+01:00",
          "tree_id": "241a8abb542b85681206d7b013d10002e1237c62",
          "url": "https://github.com/ComPWA/tensorwaves/commit/542ace413be3f05ccb7d379822b9dc4c856da9b8"
        },
        "date": 1645645639533,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2346457143536628,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.261744147999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19632744367068367,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.0935314050000216 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20360344615063883,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.9115082229999985 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.40599462348222215,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.463086804999989 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.50247875395384,
            "unit": "iter/sec",
            "range": "stddev: 0.0006703298789692289",
            "extra": "mean: 54.04681249999044 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.57420155091742,
            "unit": "iter/sec",
            "range": "stddev: 0.0001415441985165628",
            "extra": "mean: 7.32202706399994 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.6885951115674733,
            "unit": "iter/sec",
            "range": "stddev: 0.08992579272492919",
            "extra": "mean: 271.10592779998797 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 84.98830028464633,
            "unit": "iter/sec",
            "range": "stddev: 0.0001707501587237152",
            "extra": "mean: 11.766325443040497 msec\nrounds: 79"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.75659107328843,
            "unit": "iter/sec",
            "range": "stddev: 0.0013939343628934428",
            "extra": "mean: 128.92261439999402 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.007483936742725,
            "unit": "iter/sec",
            "range": "stddev: 0.00042600698067649535",
            "extra": "mean: 99.92521659999625 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.876231364822951,
            "unit": "iter/sec",
            "range": "stddev: 0.0006394640188755832",
            "extra": "mean: 101.2531970000003 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.5121054643795089,
            "unit": "iter/sec",
            "range": "stddev: 0.0036501008845761613",
            "extra": "mean: 661.3295325999957 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.2018623197828004,
            "unit": "iter/sec",
            "range": "stddev: 0.0006306679412511632",
            "extra": "mean: 138.85297379999884 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.545062960982447,
            "unit": "iter/sec",
            "range": "stddev: 0.0003521180514314997",
            "extra": "mean: 104.7662025999955 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.3736174265798,
            "unit": "iter/sec",
            "range": "stddev: 0.0003041150643673374",
            "extra": "mean: 106.6823996000096 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.696016042813872,
            "unit": "iter/sec",
            "range": "stddev: 0.0031794323618215925",
            "extra": "mean: 589.6170641999902 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fab2a7c44d202c1611eb9857530046d3008a63d8",
          "message": "chore: add support for AmpForm v0.12.4 (#417)\n\n* ci: constrain ampform to v0.12.4\r\n* ci: ignore numpy invalid value\r\n* ci: run linkcheck on Python 3.8\r\n* ci: update pip constraints and pre-commit config\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-03-03T10:55:49+01:00",
          "tree_id": "11df2635b33802e7e0e6166610f03dc143c4fda1",
          "url": "https://github.com/ComPWA/tensorwaves/commit/fab2a7c44d202c1611eb9857530046d3008a63d8"
        },
        "date": 1646301571573,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2620569343171181,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.815964658999974 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25468199536065583,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.926465231999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2585813602773509,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8672547740000027 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5039218808851045,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9844345680000401 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.87058536675182,
            "unit": "iter/sec",
            "range": "stddev: 0.0005847078226521392",
            "extra": "mean: 55.95787599999369 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.77974023199556,
            "unit": "iter/sec",
            "range": "stddev: 0.0001184989702600447",
            "extra": "mean: 7.205662716534259 msec\nrounds: 127"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.268643011500175,
            "unit": "iter/sec",
            "range": "stddev: 0.00025445443712259",
            "extra": "mean: 234.26648639998575 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 83.44860291842004,
            "unit": "iter/sec",
            "range": "stddev: 0.00017095429502500547",
            "extra": "mean: 11.983424108101694 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.474595952456615,
            "unit": "iter/sec",
            "range": "stddev: 0.00021645254037839926",
            "extra": "mean: 133.78649579999546 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.513328209923081,
            "unit": "iter/sec",
            "range": "stddev: 0.000579861204658728",
            "extra": "mean: 105.11568380001108 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.3319507662557,
            "unit": "iter/sec",
            "range": "stddev: 0.0006579877645720066",
            "extra": "mean: 107.15873079999483 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2989242470712203,
            "unit": "iter/sec",
            "range": "stddev: 0.00105819108609057",
            "extra": "mean: 769.8678365999967 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.712150570559811,
            "unit": "iter/sec",
            "range": "stddev: 0.00025579995545568914",
            "extra": "mean: 148.9835469999889 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.803139312940939,
            "unit": "iter/sec",
            "range": "stddev: 0.00046668693004575017",
            "extra": "mean: 113.595839444454 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.837736896949066,
            "unit": "iter/sec",
            "range": "stddev: 0.00021406725830634072",
            "extra": "mean: 113.15113944444495 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4796062009630417,
            "unit": "iter/sec",
            "range": "stddev: 0.0021675631656118276",
            "extra": "mean: 675.8555076000107 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4a6510d78430c85790d8615f8483ed4481ebb8a6",
          "message": "ci: update pip constraints and pre-commit config (#418)\n\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-03-07T12:50:01+01:00",
          "tree_id": "9dda25c04977916a7a6ec8b202b3eda74d5700ac",
          "url": "https://github.com/ComPWA/tensorwaves/commit/4a6510d78430c85790d8615f8483ed4481ebb8a6"
        },
        "date": 1646654019740,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2793780712010412,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5793789960000026 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2615067729040766,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8239927360000365 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2731940809684097,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6604014129999882 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4705502626629402,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.1251714840000204 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.19754652807483,
            "unit": "iter/sec",
            "range": "stddev: 0.0007911807871879318",
            "extra": "mean: 54.95246287499356 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 135.98953324528193,
            "unit": "iter/sec",
            "range": "stddev: 0.0001467730540437506",
            "extra": "mean: 7.353507112906384 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.330342596076379,
            "unit": "iter/sec",
            "range": "stddev: 0.001003649198478353",
            "extra": "mean: 230.92861080000375 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 86.23783303996501,
            "unit": "iter/sec",
            "range": "stddev: 0.00014518165800078285",
            "extra": "mean: 11.595838679487368 msec\nrounds: 78"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.326576995547934,
            "unit": "iter/sec",
            "range": "stddev: 0.004153903760481727",
            "extra": "mean: 136.48938659999885 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.189333659169531,
            "unit": "iter/sec",
            "range": "stddev: 0.0005530452374875335",
            "extra": "mean: 98.14184454545614 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.061043856147649,
            "unit": "iter/sec",
            "range": "stddev: 0.0007183856144043757",
            "extra": "mean: 99.3932651818196 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.460478181610929,
            "unit": "iter/sec",
            "range": "stddev: 0.002424876368237118",
            "extra": "mean: 684.7072503999925 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.933869407604193,
            "unit": "iter/sec",
            "range": "stddev: 0.0012914113360936578",
            "extra": "mean: 144.21961839998403 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.356554430117734,
            "unit": "iter/sec",
            "range": "stddev: 0.0005153570352971259",
            "extra": "mean: 106.87695000000303 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.97528034211752,
            "unit": "iter/sec",
            "range": "stddev: 0.0012484449593461553",
            "extra": "mean: 111.41713260001325 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6839528672045985,
            "unit": "iter/sec",
            "range": "stddev: 0.0015715581403969385",
            "extra": "mean: 593.8408488000164 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b2e43a647e95f4f52694d3db4d9b989b725854a0",
          "message": "build: allow AmpForm v0.13.x (#419)\n\n* chore: update parameter names to AmpForm v0.13.x\r\n* ci: update pip constraints and pre-commit config\r\n* fix: update SymPy type hints\r\n* fix: update vscode and gitpod badges\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-03-10T13:50:52+01:00",
          "tree_id": "b4e1119a97498d924e86ca41e0678b116ce08ad8",
          "url": "https://github.com/ComPWA/tensorwaves/commit/b2e43a647e95f4f52694d3db4d9b989b725854a0"
        },
        "date": 1646916885338,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.26411836032008834,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7861813119999965 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2443903360167314,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.091814825000029 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2533584489759588,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9469771149999815 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4515872621884556,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.2144114410000384 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.356024145691148,
            "unit": "iter/sec",
            "range": "stddev: 0.0018132713152076573",
            "extra": "mean: 54.478028142861085 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 121.65342070495498,
            "unit": "iter/sec",
            "range": "stddev: 0.0004671427046270366",
            "extra": "mean: 8.220073009087772 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.3877331754977495,
            "unit": "iter/sec",
            "range": "stddev: 0.0997486934409072",
            "extra": "mean: 295.18263340000885 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 78.15028052667556,
            "unit": "iter/sec",
            "range": "stddev: 0.0007459198454520863",
            "extra": "mean: 12.795859378376297 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.8432003315239225,
            "unit": "iter/sec",
            "range": "stddev: 0.0035391382574311184",
            "extra": "mean: 146.13045820000252 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.592637287035787,
            "unit": "iter/sec",
            "range": "stddev: 0.0028837322217249235",
            "extra": "mean: 116.3787050000072 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.602903564071225,
            "unit": "iter/sec",
            "range": "stddev: 0.005088994723637663",
            "extra": "mean: 116.23982444442997 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2718011407544463,
            "unit": "iter/sec",
            "range": "stddev: 0.003451523412596246",
            "extra": "mean: 786.2864467999998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.803905912280448,
            "unit": "iter/sec",
            "range": "stddev: 0.004384452076393817",
            "extra": "mean: 146.97440159998223 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.369612527030617,
            "unit": "iter/sec",
            "range": "stddev: 0.004200515854121075",
            "extra": "mean: 119.47984411110862 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.476491295744578,
            "unit": "iter/sec",
            "range": "stddev: 0.0026732164107350423",
            "extra": "mean: 117.97334122221376 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4510750503689136,
            "unit": "iter/sec",
            "range": "stddev: 0.01968205085557503",
            "extra": "mean: 689.1442312000095 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "280d0584ee09c2e2f4660f63ed64793b00204d81",
          "message": "ci: update pip constraints and pre-commit config (#420)\n\n* ci: update pip constraints and pre-commit config\r\n* docs: add reference to phasespace\r\n* docs: remove version from zenodo config\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-03-21T11:49:35+01:00",
          "tree_id": "94078d6b0439e288b54a39ba11ad3f6cf56796f2",
          "url": "https://github.com/ComPWA/tensorwaves/commit/280d0584ee09c2e2f4660f63ed64793b00204d81"
        },
        "date": 1647859999950,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2822624848832202,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.542801660000009 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2496029153955998,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.006363460999978 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2860184731431914,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.496277667000072 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.48078192826617616,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0799450670000397 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.970513088094513,
            "unit": "iter/sec",
            "range": "stddev: 0.0011243513832101843",
            "extra": "mean: 55.646713874992315 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 130.0835059574374,
            "unit": "iter/sec",
            "range": "stddev: 0.0001104394725847085",
            "extra": "mean: 7.687369683341672 msec\nrounds: 120"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.4131774847729117,
            "unit": "iter/sec",
            "range": "stddev: 0.10989142204312147",
            "extra": "mean: 292.9821272000254 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 81.48060216793064,
            "unit": "iter/sec",
            "range": "stddev: 0.0001515692354896181",
            "extra": "mean: 12.272859716218235 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.499851892923542,
            "unit": "iter/sec",
            "range": "stddev: 0.0003164526283256128",
            "extra": "mean: 133.3359664000227 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.450589694067999,
            "unit": "iter/sec",
            "range": "stddev: 0.001268557331397213",
            "extra": "mean: 105.81350290000273 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.37468240626621,
            "unit": "iter/sec",
            "range": "stddev: 0.0011248562564799532",
            "extra": "mean: 106.6702803000112 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3479462293335873,
            "unit": "iter/sec",
            "range": "stddev: 0.0028297681013569258",
            "extra": "mean: 741.869355200015 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.077120550361416,
            "unit": "iter/sec",
            "range": "stddev: 0.00018993951477017549",
            "extra": "mean: 141.30040500001542 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.856461625986784,
            "unit": "iter/sec",
            "range": "stddev: 0.0005264710227590039",
            "extra": "mean: 112.91191022222493 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.831738478044901,
            "unit": "iter/sec",
            "range": "stddev: 0.0003290263812368814",
            "extra": "mean: 113.22799044445573 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.55571165365769,
            "unit": "iter/sec",
            "range": "stddev: 0.001819534199101898",
            "extra": "mean: 642.7926394000224 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "005c6d2cf19ff0dc4c4d81e2888a452b511ed525",
          "message": "fix: compute default weights from observed values (#421)",
          "timestamp": "2022-03-30T12:15:59Z",
          "tree_id": "1551d0488a35c5a1eb01bc27251d625fec066cd7",
          "url": "https://github.com/ComPWA/tensorwaves/commit/005c6d2cf19ff0dc4c4d81e2888a452b511ed525"
        },
        "date": 1648642820607,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.23886270550356442,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.186505373000045 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2107615918829699,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.744697508999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.24276071082923642,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.119282715000054 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.38192871033424336,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.6182896779999965 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.609061406221253,
            "unit": "iter/sec",
            "range": "stddev: 0.00046765092491463794",
            "extra": "mean: 64.06535114285816 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 106.35049995952909,
            "unit": "iter/sec",
            "range": "stddev: 0.0001242509133561921",
            "extra": "mean: 9.402870700001813 msec\nrounds: 100"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.920266996437086,
            "unit": "iter/sec",
            "range": "stddev: 0.11626954335561679",
            "extra": "mean: 342.434442199999 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 67.12122931048593,
            "unit": "iter/sec",
            "range": "stddev: 0.00017396797493110153",
            "extra": "mean: 14.898416049179485 msec\nrounds: 61"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.2004835506620175,
            "unit": "iter/sec",
            "range": "stddev: 0.0004338271008297993",
            "extra": "mean: 161.27774420000378 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 7.542889065224932,
            "unit": "iter/sec",
            "range": "stddev: 0.0013806415245572754",
            "extra": "mean: 132.5751965000137 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 7.55125077081966,
            "unit": "iter/sec",
            "range": "stddev: 0.001400446009107174",
            "extra": "mean: 132.42839237498316 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0969209105134192,
            "unit": "iter/sec",
            "range": "stddev: 0.0012526187383839278",
            "extra": "mean: 911.6427541999769 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.746810051768867,
            "unit": "iter/sec",
            "range": "stddev: 0.0004040964934357151",
            "extra": "mean: 174.00957940000126 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.072518815169793,
            "unit": "iter/sec",
            "range": "stddev: 0.0003830772702056592",
            "extra": "mean: 141.39234212500185 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.109832384057504,
            "unit": "iter/sec",
            "range": "stddev: 0.001154269658920176",
            "extra": "mean: 140.6502918749979 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2667462983419135,
            "unit": "iter/sec",
            "range": "stddev: 0.0007391831609182239",
            "extra": "mean: 789.4240553999907 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "01daba1012af63de3d1d37f058ef0bcb71172c68",
          "message": "ci: update pip constraints and pre-commit config (#422)\n\n* ci: address pylint v2.13 issues\r\n  https://pylint.pycqa.org/en/latest/whatsnew/changelog.html#what-s-new-in-pylint-2-13-0\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-04-04T11:11:44+02:00",
          "tree_id": "0965512c10ad37a26d5d7c017d26849ae9b7be5e",
          "url": "https://github.com/ComPWA/tensorwaves/commit/01daba1012af63de3d1d37f058ef0bcb71172c68"
        },
        "date": 1649063897582,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.27804089444104174,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5965932350000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2566172820077215,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.896853680999982 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2823631157891892,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5415390470000148 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4800064453729446,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0833053590000077 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.39019534482937,
            "unit": "iter/sec",
            "range": "stddev: 0.000590805652622785",
            "extra": "mean: 57.50366687498598 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 129.39310453163455,
            "unit": "iter/sec",
            "range": "stddev: 0.00011965075845808207",
            "extra": "mean: 7.72838710084057 msec\nrounds: 119"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.4359062205405158,
            "unit": "iter/sec",
            "range": "stddev: 0.10860171233994348",
            "extra": "mean: 291.04403200000206 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 81.44262051962157,
            "unit": "iter/sec",
            "range": "stddev: 0.000156393853518588",
            "extra": "mean: 12.27858329729303 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.532368735179566,
            "unit": "iter/sec",
            "range": "stddev: 0.0003502420617574124",
            "extra": "mean: 132.76036200001045 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.356645285316066,
            "unit": "iter/sec",
            "range": "stddev: 0.0005161577007308458",
            "extra": "mean: 106.87591220000172 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.378836973062825,
            "unit": "iter/sec",
            "range": "stddev: 0.0003957899028732041",
            "extra": "mean: 106.62302830000385 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.339278313338091,
            "unit": "iter/sec",
            "range": "stddev: 0.000673696353770116",
            "extra": "mean: 746.6707928000005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.881144452567229,
            "unit": "iter/sec",
            "range": "stddev: 0.00023157627741642967",
            "extra": "mean: 145.3246632000173 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.946969966959907,
            "unit": "iter/sec",
            "range": "stddev: 0.00041471364873205927",
            "extra": "mean: 111.76968333333865 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.950185006589685,
            "unit": "iter/sec",
            "range": "stddev: 0.0009443032804525175",
            "extra": "mean: 111.72953399999415 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5208879787687257,
            "unit": "iter/sec",
            "range": "stddev: 0.0010788874296587593",
            "extra": "mean: 657.5106213999902 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c82dc3da15ba2aa638567975f784f88340da3c85",
          "message": "build: allow installing AmpForm v0.14.x (#424)\n\n* ci: ignore PytestRemovedIn8Warning\r\n  https://github.com/ComPWA/tensorwaves/runs/5873479487?check_suite_focus=true#step:8:36\r\n* ci: remove pip-tools from dev requirements\r\n  Has been outsourced to:\r\n  https://github.com/ComPWA/update-pip-constraints\r\n* ci: update pip constraints and pre-commit config\r\n* fix: do not format Jupyter notebooks with prettier\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-04-08T12:01:53+02:00",
          "tree_id": "b3a7d9ed58e12b5c708a0d8309dfbd7a4e458494",
          "url": "https://github.com/ComPWA/tensorwaves/commit/c82dc3da15ba2aa638567975f784f88340da3c85"
        },
        "date": 1649412372122,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.22635809118798825,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.417778903999988 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.218855212831175,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.569230894999976 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.22706050408888134,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.404112481000027 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4126794372131588,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.423188338999978 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.301214116147923,
            "unit": "iter/sec",
            "range": "stddev: 0.0011443718444338487",
            "extra": "mean: 65.35429099999745 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 109.80370766884488,
            "unit": "iter/sec",
            "range": "stddev: 0.0001901924677803992",
            "extra": "mean: 9.107160598036296 msec\nrounds: 102"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.974295373796148,
            "unit": "iter/sec",
            "range": "stddev: 0.11606538250793652",
            "extra": "mean: 336.21408579998615 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 68.72119919766213,
            "unit": "iter/sec",
            "range": "stddev: 0.00018251636324439848",
            "extra": "mean: 14.551550492064457 msec\nrounds: 63"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.326788725082635,
            "unit": "iter/sec",
            "range": "stddev: 0.0003081024855828237",
            "extra": "mean: 158.0580675999954 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 7.621093091400412,
            "unit": "iter/sec",
            "range": "stddev: 0.0011085045106593602",
            "extra": "mean: 131.2147730000035 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 7.576724878141273,
            "unit": "iter/sec",
            "range": "stddev: 0.0025831890111826477",
            "extra": "mean: 131.98314787501175 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0183753924576773,
            "unit": "iter/sec",
            "range": "stddev: 0.13780888821430282",
            "extra": "mean: 981.9561700000122 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.852340370285251,
            "unit": "iter/sec",
            "range": "stddev: 0.0003496455408915178",
            "extra": "mean: 170.87181140000212 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.081828273680967,
            "unit": "iter/sec",
            "range": "stddev: 0.001395995714467556",
            "extra": "mean: 141.20647400000053 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.016819582300212,
            "unit": "iter/sec",
            "range": "stddev: 0.0024954526218223974",
            "extra": "mean: 142.51470887501227 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2687053981399692,
            "unit": "iter/sec",
            "range": "stddev: 0.0026112044690689835",
            "extra": "mean: 788.2050486000026 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "022c7f140ece829095899a62ada3ccc649dcdfcd",
          "message": "docs: use Chew-Mandelstam in analytic continuation (#425)\n\n* chore: bump python kernel version\r\n* docs: simplify distribution plotting\r\n  - No need to generate hit-and-miss data\r\n  - Removed need to import pandas\r\n  - Merged plotting cells\r\n* docs: reduce coupling for sub-threshold resonance\r\n* docs: set all resonances in 02 decay chain\r\n* docs: switch to Chew-Mandelstam phase space factor",
          "timestamp": "2022-04-08T12:17:52+02:00",
          "tree_id": "83153ab23118000302092cbc6d1a920d552cae67",
          "url": "https://github.com/ComPWA/tensorwaves/commit/022c7f140ece829095899a62ada3ccc649dcdfcd"
        },
        "date": 1649413291502,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.27953411414440293,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5773808969999834 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2721317652515392,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6746904540000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.28436052042894694,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.516662574999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4547641961237901,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.1989418 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.45722859927415,
            "unit": "iter/sec",
            "range": "stddev: 0.0006682356716299864",
            "extra": "mean: 54.179314875003826 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 127.37661693457737,
            "unit": "iter/sec",
            "range": "stddev: 0.00010875255738536511",
            "extra": "mean: 7.85073449166589 msec\nrounds: 120"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.6996960203218996,
            "unit": "iter/sec",
            "range": "stddev: 0.0812205792947336",
            "extra": "mean: 270.29247660000806 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.59803334257518,
            "unit": "iter/sec",
            "range": "stddev: 0.0001463445116177673",
            "extra": "mean: 12.106825786668576 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.587941464385572,
            "unit": "iter/sec",
            "range": "stddev: 0.0016755686460174044",
            "extra": "mean: 131.78804880000143 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.880465330119042,
            "unit": "iter/sec",
            "range": "stddev: 0.0006616038128460741",
            "extra": "mean: 101.2098080999948 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.816085796877559,
            "unit": "iter/sec",
            "range": "stddev: 0.0007812539791430198",
            "extra": "mean: 101.87360019999971 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.476698647953604,
            "unit": "iter/sec",
            "range": "stddev: 0.004393204313621106",
            "extra": "mean: 677.1862365999937 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.102792270295687,
            "unit": "iter/sec",
            "range": "stddev: 0.00069525363197137",
            "extra": "mean: 140.78970099999424 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.1618742788134,
            "unit": "iter/sec",
            "range": "stddev: 0.0004839560412984773",
            "extra": "mean: 109.1479722999992 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.139258418327122,
            "unit": "iter/sec",
            "range": "stddev: 0.0009583750070128919",
            "extra": "mean: 109.41806809999832 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6474317559508722,
            "unit": "iter/sec",
            "range": "stddev: 0.0031213926785280664",
            "extra": "mean: 607.0054169999992 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "16315bbd0afd47cbd61d357b86154f11884c5805",
          "message": "build: remove version limit from AmpForm (#426)\n\n* ci: simplify release drafter template\r\n* ci: update pip constraints and pre-commit config\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-04-09T20:44:37+02:00",
          "tree_id": "e812cc524fa204f1657f6c9195076796cb885885",
          "url": "https://github.com/ComPWA/tensorwaves/commit/16315bbd0afd47cbd61d357b86154f11884c5805"
        },
        "date": 1649530099424,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.26882299360658635,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7199198869999464 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26067379529834844,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8362122240000076 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2724146671393322,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.670874298000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4922040845897628,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0316775729999677 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.736289758228367,
            "unit": "iter/sec",
            "range": "stddev: 0.0012027334038042599",
            "extra": "mean: 56.381577749995415 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 129.25231179996086,
            "unit": "iter/sec",
            "range": "stddev: 0.00016577476680684265",
            "extra": "mean: 7.73680552459026 msec\nrounds: 122"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.532491749249069,
            "unit": "iter/sec",
            "range": "stddev: 0.09724736399234214",
            "extra": "mean: 283.0862946000025 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 80.71117346027457,
            "unit": "iter/sec",
            "range": "stddev: 0.0002467829153881476",
            "extra": "mean: 12.389858270270254 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.481181868792043,
            "unit": "iter/sec",
            "range": "stddev: 0.0006147828512648585",
            "extra": "mean: 133.6687194000092 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.191349872837218,
            "unit": "iter/sec",
            "range": "stddev: 0.0002837141473100296",
            "extra": "mean: 108.79794740000648 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.252896744702772,
            "unit": "iter/sec",
            "range": "stddev: 0.000269357443380206",
            "extra": "mean: 108.07426339999893 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3207785362999946,
            "unit": "iter/sec",
            "range": "stddev: 0.005425351260026882",
            "extra": "mean: 757.1292025999924 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.0345263218410325,
            "unit": "iter/sec",
            "range": "stddev: 0.0011076604444854152",
            "extra": "mean: 142.15598240000418 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.708029438828634,
            "unit": "iter/sec",
            "range": "stddev: 0.0015480907387749104",
            "extra": "mean: 114.83654333333486 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.81259396758734,
            "unit": "iter/sec",
            "range": "stddev: 0.0009619171841099052",
            "extra": "mean: 113.47396733334057 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.515882220911404,
            "unit": "iter/sec",
            "range": "stddev: 0.003479819306768461",
            "extra": "mean: 659.681857999999 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a6406d601d7063942d5df785d4b4d3e3b59df539",
          "message": "feat: lambdify Sympy.indexed expressions (#427)\n\n* ci: further simplify release template\r\n* ci: ignore DeprecationWarning Pillow\r\n* fix: restore link to documentation in release template\r\n* style: use Prettier style (dash) for itemize in release template\r\n* test: lambdify expression with sympy.Indexed symbols",
          "timestamp": "2022-04-09T21:19:56+02:00",
          "tree_id": "15b0926f1484ad80cd71c57a21b52d43390987cd",
          "url": "https://github.com/ComPWA/tensorwaves/commit/a6406d601d7063942d5df785d4b4d3e3b59df539"
        },
        "date": 1649532221319,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.25799110783854656,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.876102584999984 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2551316978585778,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9195443310000257 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.25712322516017466,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8891858149999905 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4932335182712763,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.02743723399999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.115654122988747,
            "unit": "iter/sec",
            "range": "stddev: 0.0018005130732283987",
            "extra": "mean: 58.42604628571329 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 129.90280902766068,
            "unit": "iter/sec",
            "range": "stddev: 0.00013434194846459298",
            "extra": "mean: 7.69806294017142 msec\nrounds: 117"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.311520269839483,
            "unit": "iter/sec",
            "range": "stddev: 0.12388303988781879",
            "extra": "mean: 301.97610719999375 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 81.92997610850321,
            "unit": "iter/sec",
            "range": "stddev: 0.000194524333160882",
            "extra": "mean: 12.205544874999832 msec\nrounds: 72"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.373208380593193,
            "unit": "iter/sec",
            "range": "stddev: 0.0005116573342417967",
            "extra": "mean: 135.62616820000244 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.488129962301192,
            "unit": "iter/sec",
            "range": "stddev: 0.0009656399438096386",
            "extra": "mean: 105.39484640000296 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.443983444262088,
            "unit": "iter/sec",
            "range": "stddev: 0.0012103718242149168",
            "extra": "mean: 105.88752150000573 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3437706273410548,
            "unit": "iter/sec",
            "range": "stddev: 0.00208458678002367",
            "extra": "mean: 744.1746229999978 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.9252572497611515,
            "unit": "iter/sec",
            "range": "stddev: 0.00027775176191654496",
            "extra": "mean: 144.39896799999588 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.931882241615886,
            "unit": "iter/sec",
            "range": "stddev: 0.0007369680880300629",
            "extra": "mean: 111.95848455555632 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.881295911567978,
            "unit": "iter/sec",
            "range": "stddev: 0.0006653433812089949",
            "extra": "mean: 112.59618077779503 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.533841015137533,
            "unit": "iter/sec",
            "range": "stddev: 0.0011040077749909544",
            "extra": "mean: 651.9580517999998 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ddaf4851d640db120e791c2bef5c19b9581ebd40",
          "message": "docs: improve docstrings of function module (#428)\n\n* docs: improve docstring create_(parametrized_)function\r\n* docs: remove __eq__ and __call__ from default API\r\n* docs: remove links to General index etc",
          "timestamp": "2022-04-11T10:47:41+02:00",
          "tree_id": "7b6a71d83cbd9459bd4afc4e870cdeaea8684e38",
          "url": "https://github.com/ComPWA/tensorwaves/commit/ddaf4851d640db120e791c2bef5c19b9581ebd40"
        },
        "date": 1649667129309,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21011437139189099,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.75931271799999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2088598949749773,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.787898605999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21093634673848213,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.74076666000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3987470046867685,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5078558289999933 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 13.959805930967788,
            "unit": "iter/sec",
            "range": "stddev: 0.0009750857272808941",
            "extra": "mean: 71.63423366664763 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 96.56146393785582,
            "unit": "iter/sec",
            "range": "stddev: 0.0006570011210936952",
            "extra": "mean: 10.356098170213858 msec\nrounds: 94"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.807951068374972,
            "unit": "iter/sec",
            "range": "stddev: 0.10949831878595948",
            "extra": "mean: 356.1315620000187 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 60.785462121635845,
            "unit": "iter/sec",
            "range": "stddev: 0.0005525456352257115",
            "extra": "mean: 16.45130208928793 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.4668557840145935,
            "unit": "iter/sec",
            "range": "stddev: 0.0011754468048726853",
            "extra": "mean: 182.92050119998748 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.995305345432203,
            "unit": "iter/sec",
            "range": "stddev: 0.00260715659764777",
            "extra": "mean: 200.1879626666702 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.0981269764440205,
            "unit": "iter/sec",
            "range": "stddev: 0.0012972166034829222",
            "extra": "mean: 196.1504694999784 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8401193898638426,
            "unit": "iter/sec",
            "range": "stddev: 0.011777656448467956",
            "extra": "mean: 1.1903070112000023 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.396791740718652,
            "unit": "iter/sec",
            "range": "stddev: 0.0014868460782706161",
            "extra": "mean: 185.29527320000625 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.201992555969863,
            "unit": "iter/sec",
            "range": "stddev: 0.001495395939552732",
            "extra": "mean: 192.23403133331848 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.16866331742977,
            "unit": "iter/sec",
            "range": "stddev: 0.003065025470267122",
            "extra": "mean: 193.47361949999708 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9658562199835012,
            "unit": "iter/sec",
            "range": "stddev: 0.008688671142019815",
            "extra": "mean: 1.035350789600011 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "redeboer@gmx.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a10830d95023f249e3bd97a84945e797a002f03d",
          "message": "ci: pin requirements on Read the Docs (#429)",
          "timestamp": "2022-04-11T11:54:16+02:00",
          "tree_id": "8aceb5ec3ba6944395a5a2d3bf91f01693c2282c",
          "url": "https://github.com/ComPWA/tensorwaves/commit/a10830d95023f249e3bd97a84945e797a002f03d"
        },
        "date": 1649671131588,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2032783365304882,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.919363357000009 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.20336129345926904,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.917356606999988 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.196294690156434,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.0943813060000025 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.38971889519579905,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.565952055000025 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 13.000838168369826,
            "unit": "iter/sec",
            "range": "stddev: 0.0023800115443323297",
            "extra": "mean: 76.91811766666963 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 91.18691615308856,
            "unit": "iter/sec",
            "range": "stddev: 0.001226120684281364",
            "extra": "mean: 10.96648556818345 msec\nrounds: 88"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.742500101564306,
            "unit": "iter/sec",
            "range": "stddev: 0.10965334837430625",
            "extra": "mean: 364.63079780000953 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 55.02742890088412,
            "unit": "iter/sec",
            "range": "stddev: 0.0015879023239609184",
            "extra": "mean: 18.1727552962943 msec\nrounds: 54"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.717123504653311,
            "unit": "iter/sec",
            "range": "stddev: 0.007660039132047811",
            "extra": "mean: 211.99360139999044 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.482895309423477,
            "unit": "iter/sec",
            "range": "stddev: 0.0043497969237597",
            "extra": "mean: 223.07012119999854 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.249569625898348,
            "unit": "iter/sec",
            "range": "stddev: 0.007872271748971496",
            "extra": "mean: 235.31794700001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.816082161370987,
            "unit": "iter/sec",
            "range": "stddev: 0.018386724491238276",
            "extra": "mean: 1.2253668164000033 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.9046742232,
            "unit": "iter/sec",
            "range": "stddev: 0.0026744562543423427",
            "extra": "mean: 203.88714000000618 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.584104896635105,
            "unit": "iter/sec",
            "range": "stddev: 0.003444283524761956",
            "extra": "mean: 218.14509539998426 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.482558913724798,
            "unit": "iter/sec",
            "range": "stddev: 0.003925371998438716",
            "extra": "mean: 223.0868616000066 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9533710612782839,
            "unit": "iter/sec",
            "range": "stddev: 0.011619562710817962",
            "extra": "mean: 1.0489095386000031 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "876e40e73527d99d932e8ab4ad51fab0cff69ae8",
          "message": "build!: drop Python 3.6 support (#431)\n\n* docs: update to Myst-NB configuration v0.14\r\n  https://github.com/executablebooks/MyST-NB/blob/master/CHANGELOG.md#v0140---2022-04-27\r\n* fix: write missing latex in analytic continuation notebook\r\n* style: rewrite type hints with PEP563\r\n  https://peps.python.org/pep-0563\r\n* style: run pyupgrade for Python 3.7\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-05-03T22:32:23+01:00",
          "tree_id": "9d2360fb430cdbe9d6173b717037dafa8bf24c20",
          "url": "https://github.com/ComPWA/tensorwaves/commit/876e40e73527d99d932e8ab4ad51fab0cff69ae8"
        },
        "date": 1651613771175,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2836406608872674,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5255876109999917 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26217558601416163,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8142376840000054 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.27614261005123775,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.621317260000012 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4956464293666167,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.017567243000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.952288988882625,
            "unit": "iter/sec",
            "range": "stddev: 0.0006962438734917652",
            "extra": "mean: 52.76407512499404 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 131.87075896267504,
            "unit": "iter/sec",
            "range": "stddev: 0.00010346820314080373",
            "extra": "mean: 7.583182260163088 msec\nrounds: 123"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.566570029239048,
            "unit": "iter/sec",
            "range": "stddev: 0.10106900990445401",
            "extra": "mean: 280.38142860000335 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.09490643118451,
            "unit": "iter/sec",
            "range": "stddev: 0.0001567284379301901",
            "extra": "mean: 12.181023689188843 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.532391951859383,
            "unit": "iter/sec",
            "range": "stddev: 0.0006597721521523135",
            "extra": "mean: 132.75995280000643 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.379715308926173,
            "unit": "iter/sec",
            "range": "stddev: 0.00043976494393103255",
            "extra": "mean: 106.61304389999486 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.506712118548673,
            "unit": "iter/sec",
            "range": "stddev: 0.00105212666459619",
            "extra": "mean: 105.18883790000189 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3682487470792923,
            "unit": "iter/sec",
            "range": "stddev: 0.0006586556370849839",
            "extra": "mean: 730.861257600003 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.962325330104844,
            "unit": "iter/sec",
            "range": "stddev: 0.0003196453107546847",
            "extra": "mean: 143.6301741999955 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.990845736650709,
            "unit": "iter/sec",
            "range": "stddev: 0.00025924525479655925",
            "extra": "mean: 111.22424177778436 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.864153818285617,
            "unit": "iter/sec",
            "range": "stddev: 0.0008686614360792318",
            "extra": "mean: 112.8139267999984 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5559302435432745,
            "unit": "iter/sec",
            "range": "stddev: 0.001974110132096617",
            "extra": "mean: 642.7023345999942 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d9bef492abefdd6f5f299daebcac8be7927be445",
          "message": "ci: update pip constraints and pre-commit config (#432)\n\n* ci: allow compound words in cSpell\r\n* fix: set markdown<3.3.6 for Python 3.7\r\n* style: sort cSpell config\r\n  https://results.pre-commit.ci/run/github/244342170/1652689880.iGaA0tV8Ql-GOPPlc45BXg\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-05-16T10:52:55+02:00",
          "tree_id": "b6a63023d12a6067ec6f87dadd1351f7e4c35684",
          "url": "https://github.com/ComPWA/tensorwaves/commit/d9bef492abefdd6f5f299daebcac8be7927be445"
        },
        "date": 1652691400979,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.28149853058386226,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5524164120000137 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25838631484644564,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8701740089999817 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2825721139215927,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5389196269999843 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.493235883687276,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0274275110000417 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.680079583169277,
            "unit": "iter/sec",
            "range": "stddev: 0.000833187854466429",
            "extra": "mean: 56.560831374987686 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 129.92577278364004,
            "unit": "iter/sec",
            "range": "stddev: 0.00011465496987835961",
            "extra": "mean: 7.696702344539895 msec\nrounds: 119"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.489917792222179,
            "unit": "iter/sec",
            "range": "stddev: 0.1019120395120995",
            "extra": "mean: 286.5397008000173 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 81.29942532645197,
            "unit": "iter/sec",
            "range": "stddev: 0.0002041259655601319",
            "extra": "mean: 12.300209945944541 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.588142539609458,
            "unit": "iter/sec",
            "range": "stddev: 0.00011169180604338334",
            "extra": "mean: 131.784556599996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.202588720968656,
            "unit": "iter/sec",
            "range": "stddev: 0.0004016540300847673",
            "extra": "mean: 108.66507569999726 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.249973168370996,
            "unit": "iter/sec",
            "range": "stddev: 0.0012393119721994725",
            "extra": "mean: 108.10842170000683 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.328647811475027,
            "unit": "iter/sec",
            "range": "stddev: 0.001383204757668758",
            "extra": "mean: 752.644900599978 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.7744089531755325,
            "unit": "iter/sec",
            "range": "stddev: 0.00012186211090424753",
            "extra": "mean: 147.61435380000876 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.727925766546965,
            "unit": "iter/sec",
            "range": "stddev: 0.0012077357691998407",
            "extra": "mean: 114.57476000000749 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.816498116219565,
            "unit": "iter/sec",
            "range": "stddev: 0.0011152351068614336",
            "extra": "mean: 113.42371844443733 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5500748730186011,
            "unit": "iter/sec",
            "range": "stddev: 0.00093848032400849",
            "extra": "mean: 645.130127199991 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5b51859abdc42a1133c6ac6d8fea4c890752c570",
          "message": "chore: build documentation with make (#433)\n\n* ci: add jcache tox job\r\n* ci: define dependabot for GitHub Actions\r\n* ci: pass GITHUB_REPO environment variable\r\n* ci: pass GITHUB_TOKEN environment variable\r\n* ci: run MyST-NB with cache\r\n  https://myst-nb.readthedocs.io/en/v0.15.0/computation/execute.html#notebook-execution-modes\r\n* fix: update link to scipy inventory\r\n* style: use 2 spaces indent size in ini files",
          "timestamp": "2022-05-21T17:04:46+02:00",
          "tree_id": "fdd330570212fb82e5c3907dd2ef020aa7da6887",
          "url": "https://github.com/ComPWA/tensorwaves/commit/5b51859abdc42a1133c6ac6d8fea4c890752c570"
        },
        "date": 1653145713459,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.26972490881163197,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.707481094000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2498909437812969,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.001745661000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2617823286283044,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8199675480000224 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4912757913612646,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0355165420000105 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.357038449066426,
            "unit": "iter/sec",
            "range": "stddev: 0.0011177423809431409",
            "extra": "mean: 57.613515285713184 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 130.27862438949205,
            "unit": "iter/sec",
            "range": "stddev: 0.00012502482733496058",
            "extra": "mean: 7.675856301723871 msec\nrounds: 116"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.04799174988193,
            "unit": "iter/sec",
            "range": "stddev: 0.001085149798781903",
            "extra": "mean: 247.0360765999999 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 80.55218061459212,
            "unit": "iter/sec",
            "range": "stddev: 0.0003021724251423288",
            "extra": "mean: 12.414313211265803 msec\nrounds: 71"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.464263498664965,
            "unit": "iter/sec",
            "range": "stddev: 0.00025230379522123576",
            "extra": "mean: 133.97169059999783 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.166093514658199,
            "unit": "iter/sec",
            "range": "stddev: 0.001265315955459717",
            "extra": "mean: 109.0977305000024 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.186623890512521,
            "unit": "iter/sec",
            "range": "stddev: 0.000885320323454761",
            "extra": "mean: 108.8539175999955 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3091278135086948,
            "unit": "iter/sec",
            "range": "stddev: 0.0017447833682384865",
            "extra": "mean: 763.8673548000043 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.744262008859544,
            "unit": "iter/sec",
            "range": "stddev: 0.0002649268199830288",
            "extra": "mean: 148.27419200000804 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.70253406217519,
            "unit": "iter/sec",
            "range": "stddev: 0.0014317689774106907",
            "extra": "mean: 114.90905900000017 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.69927628400783,
            "unit": "iter/sec",
            "range": "stddev: 0.0012063378027691266",
            "extra": "mean: 114.95209111111156 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5062567046140432,
            "unit": "iter/sec",
            "range": "stddev: 0.0020714967866242114",
            "extra": "mean: 663.8974598000118 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3cbe2e0288d7d70324009002cad097cad5405fba",
          "message": "ci: update pip constraints and pre-commit config (#434)\n\n* chore: replace maps with generators (flake8 C417)\r\n\r\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-05-22T00:13:11+02:00",
          "tree_id": "15fbe29b16b81d277236fdba38a6be848ad7039c",
          "url": "https://github.com/ComPWA/tensorwaves/commit/3cbe2e0288d7d70324009002cad097cad5405fba"
        },
        "date": 1653171413291,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.27917883591275994,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5819334109999943 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2555507579592032,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.913116940000009 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.27748439814835585,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6038062199999956 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4910236933304238,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.036561603000024 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.33094023089152,
            "unit": "iter/sec",
            "range": "stddev: 0.0005831616913561126",
            "extra": "mean: 57.7002739999963 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 125.5016403774388,
            "unit": "iter/sec",
            "range": "stddev: 0.0001846910304158527",
            "extra": "mean: 7.9680233421057975 msec\nrounds: 114"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.45374129810613,
            "unit": "iter/sec",
            "range": "stddev: 0.10894519519081215",
            "extra": "mean: 289.54108420001035 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 79.54612290363667,
            "unit": "iter/sec",
            "range": "stddev: 0.0002820797791043553",
            "extra": "mean: 12.571322944443372 msec\nrounds: 72"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.69944495933179,
            "unit": "iter/sec",
            "range": "stddev: 0.00021348142637117907",
            "extra": "mean: 129.87949200000344 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.022701885776861,
            "unit": "iter/sec",
            "range": "stddev: 0.0014202299598113177",
            "extra": "mean: 110.8315460999961 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.080207442746955,
            "unit": "iter/sec",
            "range": "stddev: 0.0014017733549163938",
            "extra": "mean: 110.12964255555364 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2974437747440222,
            "unit": "iter/sec",
            "range": "stddev: 0.009454795201024425",
            "extra": "mean: 770.7463085999962 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.859335044180546,
            "unit": "iter/sec",
            "range": "stddev: 0.0001755651275398837",
            "extra": "mean: 145.78672619999793 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.56358964904738,
            "unit": "iter/sec",
            "range": "stddev: 0.0011286038069933217",
            "extra": "mean: 116.77346077777567 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.570021145422794,
            "unit": "iter/sec",
            "range": "stddev: 0.0010967094402227194",
            "extra": "mean: 116.68582644444176 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.512560002555746,
            "unit": "iter/sec",
            "range": "stddev: 0.008076622390774802",
            "extra": "mean: 661.1307969999984 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5ec92142f6fb03ff6453b05ce19159b087d60a56",
          "message": "build(deps): bump actions/download-artifact (#435)",
          "timestamp": "2022-05-22T00:20:56+02:00",
          "tree_id": "15fbe29b16b81d277236fdba38a6be848ad7039c",
          "url": "https://github.com/ComPWA/tensorwaves/commit/5ec92142f6fb03ff6453b05ce19159b087d60a56"
        },
        "date": 1653171993657,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2149793281198803,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.651610035000033 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.1984817643170101,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.0382462260000125 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21552014634935102,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.63993745800002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.40397391776818853,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4754073369999787 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 12.501828184010847,
            "unit": "iter/sec",
            "range": "stddev: 0.004269534795569443",
            "extra": "mean: 79.9883013333158 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 94.37944550716207,
            "unit": "iter/sec",
            "range": "stddev: 0.001032629318611505",
            "extra": "mean: 10.59552739080369 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.73055854738302,
            "unit": "iter/sec",
            "range": "stddev: 0.13575021034235704",
            "extra": "mean: 366.2254380000036 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 56.38718167669466,
            "unit": "iter/sec",
            "range": "stddev: 0.002576420826498928",
            "extra": "mean: 17.73452707272492 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.025220152784276,
            "unit": "iter/sec",
            "range": "stddev: 0.008333839305081353",
            "extra": "mean: 198.996256800001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.935931252025476,
            "unit": "iter/sec",
            "range": "stddev: 0.003061934034332478",
            "extra": "mean: 202.59601460001022 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.938059061121817,
            "unit": "iter/sec",
            "range": "stddev: 0.0016014596745241093",
            "extra": "mean: 202.5087160000112 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8079558430685322,
            "unit": "iter/sec",
            "range": "stddev: 0.017349239096485416",
            "extra": "mean: 1.2376914018000094 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.166913097419709,
            "unit": "iter/sec",
            "range": "stddev: 0.0030346312216458383",
            "extra": "mean: 193.53915600000846 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.023179806467707,
            "unit": "iter/sec",
            "range": "stddev: 0.00935030356970747",
            "extra": "mean: 199.07708633332768 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.097499776270689,
            "unit": "iter/sec",
            "range": "stddev: 0.003838649994271393",
            "extra": "mean: 196.17460400000178 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9351328645152821,
            "unit": "iter/sec",
            "range": "stddev: 0.017713949505234527",
            "extra": "mean: 1.0693667583999855 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f1951d3355ee1ba8425e27a5dbfa2c3fa9760916",
          "message": "build(deps): bump codecov/codecov-action (#436)",
          "timestamp": "2022-05-22T00:44:19+02:00",
          "tree_id": "276fd0383db88324bcb43b0eee9993d971612f85",
          "url": "https://github.com/ComPWA/tensorwaves/commit/f1951d3355ee1ba8425e27a5dbfa2c3fa9760916"
        },
        "date": 1653173286519,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.27901616591875134,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5840217240000243 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26952735098049374,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.710198598999966 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2818516384199054,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5479658930000255 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5017474171651064,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9930346740000005 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.944576524502317,
            "unit": "iter/sec",
            "range": "stddev: 0.0005882429472718037",
            "extra": "mean: 55.72714400000223 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.44868666589318,
            "unit": "iter/sec",
            "range": "stddev: 0.0001709357486929218",
            "extra": "mean: 7.328762367999843 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.62289331851115,
            "unit": "iter/sec",
            "range": "stddev: 0.12158111686156103",
            "extra": "mean: 276.0224803999904 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 83.35745819058994,
            "unit": "iter/sec",
            "range": "stddev: 0.0002811456372479318",
            "extra": "mean: 11.99652702597508 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.980322210871605,
            "unit": "iter/sec",
            "range": "stddev: 0.00035435007126104234",
            "extra": "mean: 125.30822360000684 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.160491778399063,
            "unit": "iter/sec",
            "range": "stddev: 0.0021044853034437186",
            "extra": "mean: 109.16444490000572 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.505754565180697,
            "unit": "iter/sec",
            "range": "stddev: 0.00155117253957695",
            "extra": "mean: 105.19943400000784 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.3693158452160377,
            "unit": "iter/sec",
            "range": "stddev: 0.010529707328312609",
            "extra": "mean: 730.2917026000159 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.350266457743599,
            "unit": "iter/sec",
            "range": "stddev: 0.00044920394611990444",
            "extra": "mean: 136.04948959999774 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.915875882046386,
            "unit": "iter/sec",
            "range": "stddev: 0.0015058989290430373",
            "extra": "mean: 112.15947969998865 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.900483758737236,
            "unit": "iter/sec",
            "range": "stddev: 0.0018068117309027682",
            "extra": "mean: 112.35344359999999 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5876730034918305,
            "unit": "iter/sec",
            "range": "stddev: 0.00484517689918502",
            "extra": "mean: 629.8526194000033 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a5fab689ef7725b46fffea079b3cb2164a4b382c",
          "message": "build(deps): bump actions/setup-python (#438)",
          "timestamp": "2022-05-22T08:23:17Z",
          "tree_id": "0d6d68bbbb773758e8cfb62769474da8e6d776ac",
          "url": "https://github.com/ComPWA/tensorwaves/commit/a5fab689ef7725b46fffea079b3cb2164a4b382c"
        },
        "date": 1653208032104,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2765720414506497,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.615694466999969 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2528331614324043,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9551773759999946 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2650592821518215,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7727409199999897 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.48675043485565744,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0544408970000063 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.215625921245405,
            "unit": "iter/sec",
            "range": "stddev: 0.0001359429773423028",
            "extra": "mean: 58.08676400001949 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 123.88416041108076,
            "unit": "iter/sec",
            "range": "stddev: 0.0001824486358939732",
            "extra": "mean: 8.072056965811711 msec\nrounds: 117"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.397930352019387,
            "unit": "iter/sec",
            "range": "stddev: 0.11070085856193154",
            "extra": "mean: 294.29679140000644 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 76.33594937033412,
            "unit": "iter/sec",
            "range": "stddev: 0.0002682218698227126",
            "extra": "mean: 13.099987728568456 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.452028986734541,
            "unit": "iter/sec",
            "range": "stddev: 0.0004281471842311621",
            "extra": "mean: 134.19164120001597 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.66116486313928,
            "unit": "iter/sec",
            "range": "stddev: 0.0016527987797744273",
            "extra": "mean: 115.45791077778252 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.62247261154567,
            "unit": "iter/sec",
            "range": "stddev: 0.0019766300960412397",
            "extra": "mean: 115.97601349999991 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2826103952726648,
            "unit": "iter/sec",
            "range": "stddev: 0.00341239880386217",
            "extra": "mean: 779.659983800002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.764963050034953,
            "unit": "iter/sec",
            "range": "stddev: 0.0006283128609701794",
            "extra": "mean: 147.82046740001533 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.458697607224261,
            "unit": "iter/sec",
            "range": "stddev: 0.001057461743180065",
            "extra": "mean: 118.22150955555344 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.311125237150343,
            "unit": "iter/sec",
            "range": "stddev: 0.0026094434712434166",
            "extra": "mean: 120.3206511111211 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.477314896689282,
            "unit": "iter/sec",
            "range": "stddev: 0.0025838878450288767",
            "extra": "mean: 676.9037543999843 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9b21e214322e68d643ee8da50434023d939fa8fd",
          "message": "build(deps): bump peter-evans/create-pull-request (#437)",
          "timestamp": "2022-05-22T10:33:18+02:00",
          "tree_id": "0d6d68bbbb773758e8cfb62769474da8e6d776ac",
          "url": "https://github.com/ComPWA/tensorwaves/commit/9b21e214322e68d643ee8da50434023d939fa8fd"
        },
        "date": 1653208630210,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2705966729727179,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6955369370000426 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2494124958342608,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.009422208999979 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.25858703076348344,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8671699699999635 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.48897660720656005,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0450876079999603 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.75851000489896,
            "unit": "iter/sec",
            "range": "stddev: 0.00036549771578648943",
            "extra": "mean: 59.67117599999483 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 126.80119247021389,
            "unit": "iter/sec",
            "range": "stddev: 0.00017168105773807606",
            "extra": "mean: 7.886361165214627 msec\nrounds: 115"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.2588992361563056,
            "unit": "iter/sec",
            "range": "stddev: 0.13327454025056407",
            "extra": "mean: 306.8520771999829 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 78.96197953388237,
            "unit": "iter/sec",
            "range": "stddev: 0.0003402659186543092",
            "extra": "mean: 12.6643228285697 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.307937404108058,
            "unit": "iter/sec",
            "range": "stddev: 0.0027073656936500083",
            "extra": "mean: 136.83751579999353 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.009455432485952,
            "unit": "iter/sec",
            "range": "stddev: 0.0011539271184972005",
            "extra": "mean: 110.99449988888763 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.11675165893629,
            "unit": "iter/sec",
            "range": "stddev: 0.0010627278699798098",
            "extra": "mean: 109.68819130000043 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.31985124538402,
            "unit": "iter/sec",
            "range": "stddev: 0.004754628848196661",
            "extra": "mean: 757.6611406000097 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.79021202127395,
            "unit": "iter/sec",
            "range": "stddev: 0.000321337967827503",
            "extra": "mean: 147.27080640000167 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.343702910488513,
            "unit": "iter/sec",
            "range": "stddev: 0.00220102165411298",
            "extra": "mean: 119.85086366665125 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.590170212103581,
            "unit": "iter/sec",
            "range": "stddev: 0.0012713741118831239",
            "extra": "mean: 116.41212866667023 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5262181381333026,
            "unit": "iter/sec",
            "range": "stddev: 0.001674245115331406",
            "extra": "mean: 655.2143333999993 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d6bb2c628a721ead44418ce5bb77018a0979e425",
          "message": "ci: update pip constraints and pre-commit config (#443)\n\n* fix: ignore mypy errors\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-05-30T13:34:14+02:00",
          "tree_id": "278b33081139b3fc53015000132d3b55a283ec53",
          "url": "https://github.com/ComPWA/tensorwaves/commit/d6bb2c628a721ead44418ce5bb77018a0979e425"
        },
        "date": 1653910683044,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.270254911902754,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.700210267999978 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2554737374427292,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9142966709999882 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.28678672220379375,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.486911780000014 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.49257017249498036,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0301675899999623 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.815716948990687,
            "unit": "iter/sec",
            "range": "stddev: 0.000931303581006952",
            "extra": "mean: 53.14705800002173 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 146.54401251294163,
            "unit": "iter/sec",
            "range": "stddev: 0.0001439641097533306",
            "extra": "mean: 6.8238884881884045 msec\nrounds: 127"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.697389709639831,
            "unit": "iter/sec",
            "range": "stddev: 0.11401726477790622",
            "extra": "mean: 270.46107620000157 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 85.90353670344024,
            "unit": "iter/sec",
            "range": "stddev: 0.00025339660757036145",
            "extra": "mean: 11.640964253337339 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.930466884477008,
            "unit": "iter/sec",
            "range": "stddev: 0.0002174396137875824",
            "extra": "mean: 126.09598079999387 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 12.370622239833953,
            "unit": "iter/sec",
            "range": "stddev: 0.0015338094381042234",
            "extra": "mean: 80.83667746153914 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 12.58783455681789,
            "unit": "iter/sec",
            "range": "stddev: 0.0026568684755556003",
            "extra": "mean: 79.44178130768128 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.311239615075972,
            "unit": "iter/sec",
            "range": "stddev: 0.0054611841157526245",
            "extra": "mean: 762.6371172000177 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.249621111602219,
            "unit": "iter/sec",
            "range": "stddev: 0.00025457864402311",
            "extra": "mean: 137.9382431999943 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 11.602119781537693,
            "unit": "iter/sec",
            "range": "stddev: 0.001453656207865962",
            "extra": "mean: 86.1911459999997 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 11.723323114935635,
            "unit": "iter/sec",
            "range": "stddev: 0.0013003752531610092",
            "extra": "mean: 85.30004591667269 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.5390124878846518,
            "unit": "iter/sec",
            "range": "stddev: 0.0031916528552103788",
            "extra": "mean: 649.7673072000111 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "06579932c710fefc442ae83d75ba6445e6527a00",
          "message": "build(deps): bump pull-request-name-linter-action (#444)",
          "timestamp": "2022-06-01T13:50:14+02:00",
          "tree_id": "369b6e26a118e221ae216450fb9e40e6af51e606",
          "url": "https://github.com/ComPWA/tensorwaves/commit/06579932c710fefc442ae83d75ba6445e6527a00"
        },
        "date": 1654084438116,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2904280368922427,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.443193745000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2479343369928351,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.033325969000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.29700258392901313,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.3669740740000123 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4539019972971557,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.20311874799998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.951141686553644,
            "unit": "iter/sec",
            "range": "stddev: 0.000560500604395077",
            "extra": "mean: 55.70676324999724 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 132.50865928060216,
            "unit": "iter/sec",
            "range": "stddev: 0.0001450789588142724",
            "extra": "mean: 7.546676612902604 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.228873612473965,
            "unit": "iter/sec",
            "range": "stddev: 0.0008215753827525488",
            "extra": "mean: 236.46958779999636 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.32868041817937,
            "unit": "iter/sec",
            "range": "stddev: 0.00014990514713319008",
            "extra": "mean: 12.146435421053893 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.816348134743346,
            "unit": "iter/sec",
            "range": "stddev: 0.0004960763960783641",
            "extra": "mean: 127.93698319999864 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 12.607762009025308,
            "unit": "iter/sec",
            "range": "stddev: 0.0005864870944707293",
            "extra": "mean: 79.31621800000244 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 12.883228897891723,
            "unit": "iter/sec",
            "range": "stddev: 0.0007577484637499473",
            "extra": "mean: 77.62029285714586 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4734955425439484,
            "unit": "iter/sec",
            "range": "stddev: 0.0005358416017017141",
            "extra": "mean: 678.6583136000047 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.993230365587292,
            "unit": "iter/sec",
            "range": "stddev: 0.000520681301463338",
            "extra": "mean: 142.99543240000503 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 12.193457818751714,
            "unit": "iter/sec",
            "range": "stddev: 0.0003976746390746794",
            "extra": "mean: 82.0111911538456 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 12.117431109861478,
            "unit": "iter/sec",
            "range": "stddev: 0.0018559127880560604",
            "extra": "mean: 82.5257425384638 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6420797367068838,
            "unit": "iter/sec",
            "range": "stddev: 0.0022384962245713454",
            "extra": "mean: 608.9838256000007 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "53b0165e41d2cc6cc3b8a73be833fad47bacaa35",
          "message": "ci: update pip constraints and pre-commit config (#446)\n\n* fix: remove pylint no-self-use\r\n  https://github.com/PyCQA/pylint/releases/tag/v2.14.0\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-06-08T10:41:20+02:00",
          "tree_id": "748ca29f9eea63c44409f7f946655557abb59608",
          "url": "https://github.com/ComPWA/tensorwaves/commit/53b0165e41d2cc6cc3b8a73be833fad47bacaa35"
        },
        "date": 1654677908223,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.273919052465524,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6507135630000107 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26848213520423,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7246426069999643 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.253933877351182,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.938033044000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5444187774461348,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.83682128800001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.085505711910272,
            "unit": "iter/sec",
            "range": "stddev: 0.0010742699358710557",
            "extra": "mean: 55.292896749989495 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 141.93928409432,
            "unit": "iter/sec",
            "range": "stddev: 0.00020296992029068085",
            "extra": "mean: 7.045265913385125 msec\nrounds: 127"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5712313119162093,
            "unit": "iter/sec",
            "range": "stddev: 0.13145290198011897",
            "extra": "mean: 280.01546599999756 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 83.76838415342276,
            "unit": "iter/sec",
            "range": "stddev: 0.0003648193468154794",
            "extra": "mean: 11.937678040541984 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.98684329221249,
            "unit": "iter/sec",
            "range": "stddev: 0.001276681854147809",
            "extra": "mean: 125.20591220001052 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 11.960861895881516,
            "unit": "iter/sec",
            "range": "stddev: 0.002470810722593069",
            "extra": "mean: 83.60601507691766 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 12.20947540363101,
            "unit": "iter/sec",
            "range": "stddev: 0.002355028274685353",
            "extra": "mean: 81.90360084615979 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2952168732996119,
            "unit": "iter/sec",
            "range": "stddev: 0.01371844544938323",
            "extra": "mean: 772.0714735999877 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.3102260207846355,
            "unit": "iter/sec",
            "range": "stddev: 0.0003113467776016933",
            "extra": "mean: 136.79467599999953 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 10.947917701026055,
            "unit": "iter/sec",
            "range": "stddev: 0.0010495565926949553",
            "extra": "mean: 91.34157081819114 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 11.763170932258163,
            "unit": "iter/sec",
            "range": "stddev: 0.002598076648802866",
            "extra": "mean: 85.01109146154616 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4756881606260976,
            "unit": "iter/sec",
            "range": "stddev: 0.007249963272444384",
            "extra": "mean: 677.649944399991 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "29213ce3e67d01e31f3a65f61153f4fb480c18ec",
          "message": "fix: remove `phasespace` version limit (#445)\n\n* ci: update (downgrade) pip constraints and pre-commit config\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-06-08T10:59:36+02:00",
          "tree_id": "584b497c10aeadc66f299d7c91cb75bdc6629242",
          "url": "https://github.com/ComPWA/tensorwaves/commit/29213ce3e67d01e31f3a65f61153f4fb480c18ec"
        },
        "date": 1654679035987,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21396674047879552,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.673623562999978 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.22915265434944934,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.363903193000056 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.23697479845343142,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.219858004000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.39347996659922474,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5414254470000515 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.601611102830036,
            "unit": "iter/sec",
            "range": "stddev: 0.0032294325434902584",
            "extra": "mean: 68.48559333333999 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 102.4771713279511,
            "unit": "iter/sec",
            "range": "stddev: 0.0006483180272144808",
            "extra": "mean: 9.758270910891602 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5052202905785315,
            "unit": "iter/sec",
            "range": "stddev: 0.009178709638860647",
            "extra": "mean: 285.2887741999666 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 62.910585734496536,
            "unit": "iter/sec",
            "range": "stddev: 0.000811057131222752",
            "extra": "mean: 15.89557605170504 msec\nrounds: 58"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.553068354749062,
            "unit": "iter/sec",
            "range": "stddev: 0.002582623594263441",
            "extra": "mean: 180.0806214000204 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 5.13427122489553,
            "unit": "iter/sec",
            "range": "stddev: 0.003438887071058927",
            "extra": "mean: 194.76960920005695 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.1994283824694545,
            "unit": "iter/sec",
            "range": "stddev: 0.0030575043233700377",
            "extra": "mean: 192.3288343333335 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9987819398798629,
            "unit": "iter/sec",
            "range": "stddev: 0.013775462203985902",
            "extra": "mean: 1.0012195455999973 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.39912580767234,
            "unit": "iter/sec",
            "range": "stddev: 0.002546960230016301",
            "extra": "mean: 185.21516919997794 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.265708792532562,
            "unit": "iter/sec",
            "range": "stddev: 0.0036172215851318712",
            "extra": "mean: 189.90795720001188 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.183651027987074,
            "unit": "iter/sec",
            "range": "stddev: 0.003275596951105443",
            "extra": "mean: 192.9142210000047 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1494504035082733,
            "unit": "iter/sec",
            "range": "stddev: 0.010093331357603185",
            "extra": "mean: 869.9809899999764 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7964b710b909232f018ede39f7ee02f074f196cd",
          "message": "ci: update pip constraints and pre-commit config (#448)\n\n* ci: run linkcheck on epic branches\r\n* ci: update pip constraints and pre-commit config\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-06-13T11:31:05+02:00",
          "tree_id": "be51d15efb07ab300d2b0e5029a786b313751a3c",
          "url": "https://github.com/ComPWA/tensorwaves/commit/7964b710b909232f018ede39f7ee02f074f196cd"
        },
        "date": 1655112884161,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.30724837047769543,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.2546958620000055 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2799211711974381,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.572434323999971 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.31899080913441175,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.1348865589999946 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.45063397377512115,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.2190958919999844 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.174956371819484,
            "unit": "iter/sec",
            "range": "stddev: 0.0008350900201576985",
            "extra": "mean: 52.15135725000408 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 129.13324335918472,
            "unit": "iter/sec",
            "range": "stddev: 0.0001068433642899057",
            "extra": "mean: 7.7439393140501815 msec\nrounds: 121"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.654371648975336,
            "unit": "iter/sec",
            "range": "stddev: 0.08615402453390972",
            "extra": "mean: 273.6448550000091 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 85.67903385819407,
            "unit": "iter/sec",
            "range": "stddev: 0.00016358158898514653",
            "extra": "mean: 11.671466810130973 msec\nrounds: 79"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.005658937718236,
            "unit": "iter/sec",
            "range": "stddev: 0.00031474378354539873",
            "extra": "mean: 124.91164159998789 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.533943324688995,
            "unit": "iter/sec",
            "range": "stddev: 0.00031749987664656264",
            "extra": "mean: 104.88839359999247 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.557715382590159,
            "unit": "iter/sec",
            "range": "stddev: 0.0012318675418321727",
            "extra": "mean: 104.62751400000343 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.6516773347877742,
            "unit": "iter/sec",
            "range": "stddev: 0.0023414567527010074",
            "extra": "mean: 605.4451308000125 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.464091130869077,
            "unit": "iter/sec",
            "range": "stddev: 0.00026598735726346607",
            "extra": "mean: 133.97478439998167 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.87427395448023,
            "unit": "iter/sec",
            "range": "stddev: 0.0003027865832712867",
            "extra": "mean: 112.68527488889883 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.85546489216866,
            "unit": "iter/sec",
            "range": "stddev: 0.0008348613851349448",
            "extra": "mean: 112.92461911111535 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.8659979769865176,
            "unit": "iter/sec",
            "range": "stddev: 0.0009610278467591106",
            "extra": "mean: 535.906261599996 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2d854e4de2d43f60ce61dad84ee25e0e13ee247d",
          "message": "style: switch to black's default 88 line width (#449)\n\n* ci: add docformatter as pre-commit hook\r\n* ci: set indent size for notebooks to 4 spaces\r\n  This indent was intended for the JSON format of notebooks, but also\r\n  affects the Jupyter view in VSCode and has therefore been removed.\r\n* ci: update pip constraints and pre-commit config\r\n* docs: fix docstring for `YAMLSummary` (was referring to TF)\r\n* style: format Prettier with 88 characters as well\r\n* style: format Python source code with 88 chars line width\r\n* style: rewrap docstrings to 88 characters\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-06-27T11:38:41+02:00",
          "tree_id": "d41195842516af715cc3b730474bd3a073aacbe6",
          "url": "https://github.com/ComPWA/tensorwaves/commit/2d854e4de2d43f60ce61dad84ee25e0e13ee247d"
        },
        "date": 1656323119556,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21357912949051397,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.682105421000017 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19844101941101297,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.039280703999964 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21828629162027527,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.581139715999996 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.377047092920993,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.6521885959999736 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 12.26934159253365,
            "unit": "iter/sec",
            "range": "stddev: 0.0047283128870616335",
            "extra": "mean: 81.50396599997975 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 84.1079179530506,
            "unit": "iter/sec",
            "range": "stddev: 0.0005180814608054589",
            "extra": "mean: 11.889487034481157 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.493298708701348,
            "unit": "iter/sec",
            "range": "stddev: 0.12541615390525268",
            "extra": "mean: 401.07508840000037 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 53.397503026954155,
            "unit": "iter/sec",
            "range": "stddev: 0.0018989032244712606",
            "extra": "mean: 18.72746745283608 msec\nrounds: 53"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.993137601447283,
            "unit": "iter/sec",
            "range": "stddev: 0.003246379581705631",
            "extra": "mean: 200.27487319999864 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.623082270695998,
            "unit": "iter/sec",
            "range": "stddev: 0.0015662142358164547",
            "extra": "mean: 216.30590619998884 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.574082789870349,
            "unit": "iter/sec",
            "range": "stddev: 0.004531754228944428",
            "extra": "mean: 218.62306519999493 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8990875517914101,
            "unit": "iter/sec",
            "range": "stddev: 0.016926115514501162",
            "extra": "mean: 1.1122387335999975 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.7321246631375,
            "unit": "iter/sec",
            "range": "stddev: 0.0022991758128468576",
            "extra": "mean: 211.3215672000024 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.59647617478815,
            "unit": "iter/sec",
            "range": "stddev: 0.007356947685738046",
            "extra": "mean: 217.55796439999813 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.501642072531822,
            "unit": "iter/sec",
            "range": "stddev: 0.008734695030488487",
            "extra": "mean: 222.14116180000474 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.044680770484724,
            "unit": "iter/sec",
            "range": "stddev: 0.015835468695261323",
            "extra": "mean: 957.2302163999893 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2afbf802a333583c88a9247e7c300760db66c811",
          "message": "docs: switch to `sphinx-design` (#450)\n\n* chore: switch to importlib-metadata in `conf.py`\r\n* ci: update pip constraints and pre-commit config\r\n* ci: update to `actions/setup-python@v4`\r\n\r\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-06-27T16:27:27+02:00",
          "tree_id": "cd2859df881c0196f2b4aa506eabbb45ae95aace",
          "url": "https://github.com/ComPWA/tensorwaves/commit/2afbf802a333583c88a9247e7c300760db66c811"
        },
        "date": 1656340451457,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.31176345766215585,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.2075600120000445 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2817103901085577,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5497448269999836 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.30637621295621703,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.2639609660000133 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4568814858444599,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.188751417999981 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.301598674209387,
            "unit": "iter/sec",
            "range": "stddev: 0.0007945049044934079",
            "extra": "mean: 51.80918000000645 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 129.76779882756975,
            "unit": "iter/sec",
            "range": "stddev: 0.00009195033161887915",
            "extra": "mean: 7.706071991933531 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.7528198735440577,
            "unit": "iter/sec",
            "range": "stddev: 0.07600306710611031",
            "extra": "mean: 266.4662929999963 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 86.7319579931426,
            "unit": "iter/sec",
            "range": "stddev: 0.00013116341691161594",
            "extra": "mean: 11.529775450002688 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.823430089729526,
            "unit": "iter/sec",
            "range": "stddev: 0.0015958700421769437",
            "extra": "mean: 127.82117160001008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.424764058970856,
            "unit": "iter/sec",
            "range": "stddev: 0.00035497392996971437",
            "extra": "mean: 106.10345190001453 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.540710374313258,
            "unit": "iter/sec",
            "range": "stddev: 0.0009181899017780001",
            "extra": "mean: 104.81399820000092 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.6537928140792846,
            "unit": "iter/sec",
            "range": "stddev: 0.003193376634770309",
            "extra": "mean: 604.6706645999848 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.514021543658009,
            "unit": "iter/sec",
            "range": "stddev: 0.0004208246995820389",
            "extra": "mean: 133.08452660000967 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.861492244358228,
            "unit": "iter/sec",
            "range": "stddev: 0.0013147551192261412",
            "extra": "mean: 112.84781077776846 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.676938858806322,
            "unit": "iter/sec",
            "range": "stddev: 0.0008767001247630244",
            "extra": "mean: 115.2480173333351 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.852200105945775,
            "unit": "iter/sec",
            "range": "stddev: 0.003291206321157221",
            "extra": "mean: 539.8984681999991 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "66853113+pre-commit-ci[bot]@users.noreply.github.com",
            "name": "pre-commit-ci[bot]",
            "username": "pre-commit-ci[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ef2c96b692ce19b0270e71ae8f7f69a198c4193d",
          "message": "ci: update pip constraints and pre-commit config (#451)\n\n* ci: change pre-commit autoupdate PR title\r\n* fix: correctly render expression tree\r\n\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: Remco de Boer <29308176+redeboer@users.noreply.github.com>",
          "timestamp": "2022-07-05T13:36:10+02:00",
          "tree_id": "e1dc539dc3de372160b6acec19d9bb338d2dcbbe",
          "url": "https://github.com/ComPWA/tensorwaves/commit/ef2c96b692ce19b0270e71ae8f7f69a198c4193d"
        },
        "date": 1657021258481,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.20797801714620054,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.808200470999964 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.1999745759523221,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.000635681999995 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.19891714918508396,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.0272186390000115 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.38521094325368077,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5959802479999894 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 11.932284799238769,
            "unit": "iter/sec",
            "range": "stddev: 0.005690205220128741",
            "extra": "mean: 83.80624640000178 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 86.7278088224884,
            "unit": "iter/sec",
            "range": "stddev: 0.0008113098886244349",
            "extra": "mean: 11.53032704938697 msec\nrounds: 81"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.03998436718437,
            "unit": "iter/sec",
            "range": "stddev: 0.0054293050972553355",
            "extra": "mean: 328.9490600000022 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 41.50556789872475,
            "unit": "iter/sec",
            "range": "stddev: 0.04440949145396483",
            "extra": "mean: 24.09315305455018 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.073582635735302,
            "unit": "iter/sec",
            "range": "stddev: 0.002559994801113497",
            "extra": "mean: 197.09938160001457 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.5080638198952885,
            "unit": "iter/sec",
            "range": "stddev: 0.0029811364512268984",
            "extra": "mean: 221.82472119998238 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.528476980611688,
            "unit": "iter/sec",
            "range": "stddev: 0.0025550218083148007",
            "extra": "mean: 220.824794799978 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8582166619006095,
            "unit": "iter/sec",
            "range": "stddev: 0.09238461961371226",
            "extra": "mean: 1.1652069278000226 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.715115165282804,
            "unit": "iter/sec",
            "range": "stddev: 0.002905994817802851",
            "extra": "mean: 212.08389720000014 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.5705973934058735,
            "unit": "iter/sec",
            "range": "stddev: 0.0028766468749587314",
            "extra": "mean: 218.78978039998174 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.495248265904128,
            "unit": "iter/sec",
            "range": "stddev: 0.0016485117605524106",
            "extra": "mean: 222.45712380000668 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.0051766438339538,
            "unit": "iter/sec",
            "range": "stddev: 0.03114752638106398",
            "extra": "mean: 994.8500158000002 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e2ec33231eeeba40233ad8a38c32e33ec13f80ff",
          "message": "feat: implement logging hierarchy (#453)\n\nSee https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial\r\n\r\n* docs: add tip about disabling progress bar",
          "timestamp": "2022-07-11T09:46:00+02:00",
          "tree_id": "71c57dfd3a88bc63492bb6483f7fd7a629b48768",
          "url": "https://github.com/ComPWA/tensorwaves/commit/e2ec33231eeeba40233ad8a38c32e33ec13f80ff"
        },
        "date": 1657525784145,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.29731268441796666,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.3634622820000004 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2650194114474999,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7733085079999853 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.27131227174115446,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6857897859999866 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4816464903402553,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.076211537000006 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.849511299338008,
            "unit": "iter/sec",
            "range": "stddev: 0.001455387150275634",
            "extra": "mean: 63.09342800000195 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 127.42002748072525,
            "unit": "iter/sec",
            "range": "stddev: 0.00018929700701776205",
            "extra": "mean: 7.848059836208005 msec\nrounds: 116"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.311689238574124,
            "unit": "iter/sec",
            "range": "stddev: 0.12160494252371686",
            "extra": "mean: 301.96069980000857 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.40055707853486,
            "unit": "iter/sec",
            "range": "stddev: 0.0002808260241153744",
            "extra": "mean: 12.135840283784896 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.367327812530702,
            "unit": "iter/sec",
            "range": "stddev: 0.00047379830883127944",
            "extra": "mean: 135.73442440000463 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.94098064932039,
            "unit": "iter/sec",
            "range": "stddev: 0.0005516786618548316",
            "extra": "mean: 111.84455477778164 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.890050316671914,
            "unit": "iter/sec",
            "range": "stddev: 0.0016187267694665192",
            "extra": "mean: 112.48530259999256 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.5080351753028982,
            "unit": "iter/sec",
            "range": "stddev: 0.0007543740043456078",
            "extra": "mean: 663.1145058000016 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.815884034963239,
            "unit": "iter/sec",
            "range": "stddev: 0.00029398193595305526",
            "extra": "mean: 146.7161112000042 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.455809223662543,
            "unit": "iter/sec",
            "range": "stddev: 0.0006028138779915592",
            "extra": "mean: 118.26189233333493 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.306612184852558,
            "unit": "iter/sec",
            "range": "stddev: 0.0014623418272364528",
            "extra": "mean: 120.38602233333349 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.667033754723274,
            "unit": "iter/sec",
            "range": "stddev: 0.001955932310848565",
            "extra": "mean: 599.8678773999984 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fb4a98d095821785ca611e44dfb2010b8afdfd11",
          "message": "ci: update pip constraints and pre-commit config (#452)\n\n* build: limit `nbmake` version\r\n  https://github.com/ComPWA/tensorwaves/runs/7298085386?check_suite_focus=true#step:3:71\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-07-12T12:13:52+02:00",
          "tree_id": "1154f71298310f063f395b76e109d9a527ca4914",
          "url": "https://github.com/ComPWA/tensorwaves/commit/fb4a98d095821785ca611e44dfb2010b8afdfd11"
        },
        "date": 1657621063601,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2711104723254366,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.688533280999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2687517032256766,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.720906651000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2679458682509733,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7320971079999765 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4681005036028841,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.136293365 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.59604026929353,
            "unit": "iter/sec",
            "range": "stddev: 0.002114352754181494",
            "extra": "mean: 51.03071775000245 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 130.78225578069933,
            "unit": "iter/sec",
            "range": "stddev: 0.0006017275743519162",
            "extra": "mean: 7.646297229165692 msec\nrounds: 144"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.2896579631529645,
            "unit": "iter/sec",
            "range": "stddev: 0.011434754593378664",
            "extra": "mean: 233.11881939999353 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 67.74958395972958,
            "unit": "iter/sec",
            "range": "stddev: 0.026465490951886758",
            "extra": "mean: 14.760238241380215 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.284856521749403,
            "unit": "iter/sec",
            "range": "stddev: 0.0062755417993493835",
            "extra": "mean: 137.27106320000075 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.719541544177547,
            "unit": "iter/sec",
            "range": "stddev: 0.003362920754692985",
            "extra": "mean: 114.68492866666224 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.175381849358727,
            "unit": "iter/sec",
            "range": "stddev: 0.00336585623302345",
            "extra": "mean: 122.31844559999843 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4628733227777684,
            "unit": "iter/sec",
            "range": "stddev: 0.019551475512689143",
            "extra": "mean: 683.5861892000025 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.7239945747745775,
            "unit": "iter/sec",
            "range": "stddev: 0.005334749334061474",
            "extra": "mean: 148.72111939999968 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 7.494125823150122,
            "unit": "iter/sec",
            "range": "stddev: 0.0039347064031937376",
            "extra": "mean: 133.43784499999956 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 7.483885258582808,
            "unit": "iter/sec",
            "range": "stddev: 0.0018203020168327103",
            "extra": "mean: 133.62043450000272 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.508125875115688,
            "unit": "iter/sec",
            "range": "stddev: 0.008674433899449033",
            "extra": "mean: 663.0746255999952 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1cb94890362fd760826ee299e13b751f361b6b3c",
          "message": "docs: update Zenodo author list (#454)\n\n* ci: autoupdate `pre-commit` config (fixes cspell ref)",
          "timestamp": "2022-07-22T23:39:29+02:00",
          "tree_id": "76a2abfc16c63129f5e281beaba488bd3c7247bc",
          "url": "https://github.com/ComPWA/tensorwaves/commit/1cb94890362fd760826ee299e13b751f361b6b3c"
        },
        "date": 1658526188192,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3051888131513372,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.2766600770000025 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2717729554592931,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.679541984999986 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2801026529838187,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5701197019999995 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4635503065249132,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.1572631619999925 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.196921617704575,
            "unit": "iter/sec",
            "range": "stddev: 0.0004355295841973315",
            "extra": "mean: 61.740127142858874 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 124.00622458601113,
            "unit": "iter/sec",
            "range": "stddev: 0.00021298080942474748",
            "extra": "mean: 8.064111324560136 msec\nrounds: 114"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.3771250724997763,
            "unit": "iter/sec",
            "range": "stddev: 0.10569930718328473",
            "extra": "mean: 296.1098504000006 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 79.5342028251696,
            "unit": "iter/sec",
            "range": "stddev: 0.0003588984241849021",
            "extra": "mean: 12.573207054054201 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.186464548137002,
            "unit": "iter/sec",
            "range": "stddev: 0.0005008977228160779",
            "extra": "mean: 139.15048119999938 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.490285610009941,
            "unit": "iter/sec",
            "range": "stddev: 0.000852202451972376",
            "extra": "mean: 117.78166788888849 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.464237802467993,
            "unit": "iter/sec",
            "range": "stddev: 0.0017596539655480735",
            "extra": "mean: 118.14412866666164 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4488528390348252,
            "unit": "iter/sec",
            "range": "stddev: 0.003597262313556831",
            "extra": "mean: 690.201221999996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.601422667582223,
            "unit": "iter/sec",
            "range": "stddev: 0.0004926323012588047",
            "extra": "mean: 151.4824985999951 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.121339108439047,
            "unit": "iter/sec",
            "range": "stddev: 0.0010882322328144057",
            "extra": "mean: 123.13240300000278 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.022664483566619,
            "unit": "iter/sec",
            "range": "stddev: 0.0012072904655674418",
            "extra": "mean: 124.64686788888774 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6265210603625413,
            "unit": "iter/sec",
            "range": "stddev: 0.004450184381375157",
            "extra": "mean: 614.809131200002 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e85c831fe514e9060fb1007fab29ae958a985f90",
          "message": "ci: update pip constraints and pre-commit config (#455)\n\n* docs: widen code cells where needed\r\n* fix: use positive values for initial minuit errors\r\n* style: sort `pre-commit` hooks by name\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-07-25T12:48:50+02:00",
          "tree_id": "c45a67bfeaf5ab1b056fd480f4c1eebeecfe64d9",
          "url": "https://github.com/ComPWA/tensorwaves/commit/e85c831fe514e9060fb1007fab29ae958a985f90"
        },
        "date": 1658746486075,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.20722937689532153,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.825570655000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.20149726557007266,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.9628465040000265 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20642204755544977,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.844443759000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.3576158339697468,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.7962967660000118 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 11.78967322008513,
            "unit": "iter/sec",
            "range": "stddev: 0.0037905138525047084",
            "extra": "mean: 84.8199930000078 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 86.18537571301309,
            "unit": "iter/sec",
            "range": "stddev: 0.0011060063226120828",
            "extra": "mean: 11.602896567160993 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.9718502940353573,
            "unit": "iter/sec",
            "range": "stddev: 0.009563353080306689",
            "extra": "mean: 336.490704799985 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 55.975065683107836,
            "unit": "iter/sec",
            "range": "stddev: 0.0016347198677422264",
            "extra": "mean: 17.865097392851833 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.083275283176903,
            "unit": "iter/sec",
            "range": "stddev: 0.002111810967632279",
            "extra": "mean: 196.7235579999965 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.507552848792044,
            "unit": "iter/sec",
            "range": "stddev: 0.0035052853015338893",
            "extra": "mean: 221.8498670000031 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.470985781884078,
            "unit": "iter/sec",
            "range": "stddev: 0.002759067504316455",
            "extra": "mean: 223.66432120001036 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8831337634778619,
            "unit": "iter/sec",
            "range": "stddev: 0.013146311511295201",
            "extra": "mean: 1.132331297199994 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.549143450013665,
            "unit": "iter/sec",
            "range": "stddev: 0.009483959014862943",
            "extra": "mean: 219.82160180000392 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.477099480800213,
            "unit": "iter/sec",
            "range": "stddev: 0.0062063631139228845",
            "extra": "mean: 223.35889660000703 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.49532867201819,
            "unit": "iter/sec",
            "range": "stddev: 0.006693118803803808",
            "extra": "mean: 222.45314479998797 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.0451693649693337,
            "unit": "iter/sec",
            "range": "stddev: 0.010195001158299846",
            "extra": "mean: 956.7827316000034 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "811c17b2e0bbcc463cffef2184d2bd6236f78039",
          "message": "ci: update pip constraints and pre-commit config (#457)\n\n* fix: limit `virtualenv` for Python 3.7\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-08-08T15:04:49Z",
          "tree_id": "f22795970870e7a5fdcfeafb2c51662415448fa7",
          "url": "https://github.com/ComPWA/tensorwaves/commit/811c17b2e0bbcc463cffef2184d2bd6236f78039"
        },
        "date": 1659971288700,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.33007640246470477,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.0296016089999966 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2866496028113875,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.488579751000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.30814502583101044,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.2452251900000135 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4976351065977843,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0095045279999795 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.452129990905743,
            "unit": "iter/sec",
            "range": "stddev: 0.00191807698644223",
            "extra": "mean: 57.29959612500579 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 130.8400475498753,
            "unit": "iter/sec",
            "range": "stddev: 0.00010101391956768308",
            "extra": "mean: 7.642919876032659 msec\nrounds: 121"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.4588510989291414,
            "unit": "iter/sec",
            "range": "stddev: 0.11286263004197754",
            "extra": "mean: 289.1133417999981 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 85.34098891615307,
            "unit": "iter/sec",
            "range": "stddev: 0.0001811528077128848",
            "extra": "mean: 11.717698760000228 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.537275331347227,
            "unit": "iter/sec",
            "range": "stddev: 0.00040904964821500666",
            "extra": "mean: 132.67393799998786 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.106941422494465,
            "unit": "iter/sec",
            "range": "stddev: 0.0009787438671195999",
            "extra": "mean: 109.80635029999917 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.178997762857895,
            "unit": "iter/sec",
            "range": "stddev: 0.0011000838360662138",
            "extra": "mean: 108.94435600000065 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4949073073485515,
            "unit": "iter/sec",
            "range": "stddev: 0.0007517293298473104",
            "extra": "mean: 668.9377964000016 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.106979506163023,
            "unit": "iter/sec",
            "range": "stddev: 0.00015062727239011585",
            "extra": "mean: 140.70675160000405 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.424192532089565,
            "unit": "iter/sec",
            "range": "stddev: 0.001370651909487923",
            "extra": "mean: 118.70573900000319 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.415310275813974,
            "unit": "iter/sec",
            "range": "stddev: 0.0009655112391662792",
            "extra": "mean: 118.83103144444362 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.698623883946927,
            "unit": "iter/sec",
            "range": "stddev: 0.0008424308790602644",
            "extra": "mean: 588.7118445999931 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "642e77469f80c604b3f5f3e08d25a546012cf6bb",
          "message": "refactor: merge `WeightedDataGenerator` into `DataGenerator` (#458)\n\n* chore: upgrade Jupyter notebook kernels\r\n* docs: add link to TR-018\r\n* feat: embed weights as key to `DataSample`\r\n* feat: implement phase space weights in `UnbinnedNLL`",
          "timestamp": "2022-08-09T10:59:53+02:00",
          "tree_id": "5d434930dae76966671423993f48f6f32faadb00",
          "url": "https://github.com/ComPWA/tensorwaves/commit/642e77469f80c604b3f5f3e08d25a546012cf6bb"
        },
        "date": 1660035820072,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3005359051975464,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.3273894490000657 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26101116765696725,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.831253693000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2782622892352242,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5937316649999502 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.48375322907216634,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.067169663999948 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.749055592131324,
            "unit": "iter/sec",
            "range": "stddev: 0.0022152394639128447",
            "extra": "mean: 63.49587085714705 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 102.37855888966118,
            "unit": "iter/sec",
            "range": "stddev: 0.019497597044054",
            "extra": "mean: 9.767670211862947 msec\nrounds: 118"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.3752081707738237,
            "unit": "iter/sec",
            "range": "stddev: 0.10628891265999328",
            "extra": "mean: 296.2780217999807 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 81.92425730737835,
            "unit": "iter/sec",
            "range": "stddev: 0.0002465635503784888",
            "extra": "mean: 12.206396894731895 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.3621985348311085,
            "unit": "iter/sec",
            "range": "stddev: 0.001988061363547944",
            "extra": "mean: 135.8289911999691 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 8.523488074754226,
            "unit": "iter/sec",
            "range": "stddev: 0.0017489995641041256",
            "extra": "mean: 117.32286022220251 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 8.822778289724328,
            "unit": "iter/sec",
            "range": "stddev: 0.0012748182606975185",
            "extra": "mean: 113.34298190000709 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.4398006904407619,
            "unit": "iter/sec",
            "range": "stddev: 0.0013721272462422743",
            "extra": "mean: 694.5405753999694 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.840627733650724,
            "unit": "iter/sec",
            "range": "stddev: 0.00022283045968853643",
            "extra": "mean: 146.1854144000199 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 8.017928308636378,
            "unit": "iter/sec",
            "range": "stddev: 0.0005890984455565185",
            "extra": "mean: 124.72049655555868 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 8.106184152769561,
            "unit": "iter/sec",
            "range": "stddev: 0.0008169062215641845",
            "extra": "mean: 123.36260577775546 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.6490906714827813,
            "unit": "iter/sec",
            "range": "stddev: 0.0010462245687600593",
            "extra": "mean: 606.3947951999808 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cce6d5d073d48f14fee56c51697a0d5e7e8b8d59",
          "message": "FEAT: support Python 3.10 and TF >2.6 (#459)\n\n* FIX: update links to SymPy tutorials pages\r\n* MAINT: allow installing higher versions than TF v2.6\r\n* MAINT: ignore Python 3.10 deprecation warning for `pytest`\r\n* MAINT: ignore `mystnb.unknown_mime_type` (`application/json`)\r\n  https://github.com/ComPWA/tensorwaves/runs/8143840035?check_suite_focus=true#step:5:129\r\n* MAINT: set back `ipynb` indent size to 1\r\n* MAINT: update allowed labels and commit types\r\n* MAINT: update to `phasespace` v1.7\r\n* MAINT: update pip constraints and pre-commit config\r\n* MAINT: update workflows with repo-maintenance\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-09-01T22:24:33+02:00",
          "tree_id": "6b49c875c949e2ec494483f89c0ac38447d81930",
          "url": "https://github.com/ComPWA/tensorwaves/commit/cce6d5d073d48f14fee56c51697a0d5e7e8b8d59"
        },
        "date": 1662064151339,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21619309902703496,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.6254945439999915 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.20077776177045895,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.980631276999986 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20926154986369855,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.778708752999989 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.40632757054895047,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4610685380000064 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 11.929209660175587,
            "unit": "iter/sec",
            "range": "stddev: 0.003115112634373335",
            "extra": "mean: 83.82785016667071 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 91.19069261150776,
            "unit": "iter/sec",
            "range": "stddev: 0.0009450651095278397",
            "extra": "mean: 10.96603141573031 msec\nrounds: 89"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.134081693162826,
            "unit": "iter/sec",
            "range": "stddev: 0.007842296579964134",
            "extra": "mean: 319.0727293999885 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 57.41429714624976,
            "unit": "iter/sec",
            "range": "stddev: 0.0010906769702373157",
            "extra": "mean: 17.41726450909482 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.390707405638258,
            "unit": "iter/sec",
            "range": "stddev: 0.00360172275798418",
            "extra": "mean: 185.50441060000367 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.808548432752165,
            "unit": "iter/sec",
            "range": "stddev: 0.008321286132961911",
            "extra": "mean: 207.96296719998963 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.74691938389464,
            "unit": "iter/sec",
            "range": "stddev: 0.005935833578668477",
            "extra": "mean: 210.66294140001673 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8005681548936325,
            "unit": "iter/sec",
            "range": "stddev: 0.02729203335809622",
            "extra": "mean: 1.249112887999979 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.121900032492532,
            "unit": "iter/sec",
            "range": "stddev: 0.004420864498552644",
            "extra": "mean: 195.2400463999993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.079022840294702,
            "unit": "iter/sec",
            "range": "stddev: 0.0017279646671333174",
            "extra": "mean: 196.8882659999963 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.9551193729581335,
            "unit": "iter/sec",
            "range": "stddev: 0.004490528357651594",
            "extra": "mean: 201.81148520000534 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9157145472923409,
            "unit": "iter/sec",
            "range": "stddev: 0.028888329921571143",
            "extra": "mean: 1.0920433697999896 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a51df6e1b35cbd7fb80bb22b154482e45615c4f8",
          "message": "MAINT: update pip constraints and pre-commit config (#460)\n\n* DX: show notebook traceback in Sphinx log\r\n* MAINT: ignore pylint `too-many-ancestors`\r\n* MAINT: ignore `RuntimeWarning` zero in divide\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-09-17T20:02:12+02:00",
          "tree_id": "07d6a27a9d3d8a778a50123d4d5692a732d86a0a",
          "url": "https://github.com/ComPWA/tensorwaves/commit/a51df6e1b35cbd7fb80bb22b154482e45615c4f8"
        },
        "date": 1663437981724,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.24441795201521543,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.091352503999985 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.21198368204816118,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.717344233000006 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.22732234941671695,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.399039525000006 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.39512150901108445,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.530867029999996 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.677752224953627,
            "unit": "iter/sec",
            "range": "stddev: 0.0014429531904652086",
            "extra": "mean: 68.13032299999595 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 118.81946079788257,
            "unit": "iter/sec",
            "range": "stddev: 0.0002340275155518337",
            "extra": "mean: 8.416129759257588 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.0134805636202526,
            "unit": "iter/sec",
            "range": "stddev: 0.11422035224194044",
            "extra": "mean: 331.8421933999957 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 64.23841887272751,
            "unit": "iter/sec",
            "range": "stddev: 0.00021988726285027162",
            "extra": "mean: 15.56700830357067 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.339431790642956,
            "unit": "iter/sec",
            "range": "stddev: 0.0002812568551299211",
            "extra": "mean: 157.74284399999488 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 12.75519999504486,
            "unit": "iter/sec",
            "range": "stddev: 0.0012092691083204475",
            "extra": "mean: 78.39939792308078 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 12.765163224736785,
            "unit": "iter/sec",
            "range": "stddev: 0.001264398607055823",
            "extra": "mean: 78.33820707142739 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8652209954388336,
            "unit": "iter/sec",
            "range": "stddev: 0.00934363885966153",
            "extra": "mean: 1.1557740799999976 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.008796202250816,
            "unit": "iter/sec",
            "range": "stddev: 0.0005979891979084743",
            "extra": "mean: 166.42268540001623 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 12.945184986364376,
            "unit": "iter/sec",
            "range": "stddev: 0.0007925714495297734",
            "extra": "mean: 77.24879953846434 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 12.804609413539653,
            "unit": "iter/sec",
            "range": "stddev: 0.001691392283287313",
            "extra": "mean: 78.0968765000044 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9951775410058583,
            "unit": "iter/sec",
            "range": "stddev: 0.0049543713769584635",
            "extra": "mean: 1.0048458277999999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7bf66e3322570cbbccf4101322394a539ce602cc",
          "message": "MAINT: update pip constraints and pre-commit config (#461)\n\n* MAINT: lower constraint upgrade job frequency to monthly\r\n* MAINT: remove `mystnb.unknown_mime_type` warning suppression\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-09-19T13:44:08+02:00",
          "tree_id": "d4061dff016669f75ddfd4d98f280a41b0e798cc",
          "url": "https://github.com/ComPWA/tensorwaves/commit/7bf66e3322570cbbccf4101322394a539ce602cc"
        },
        "date": 1663588067888,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3028588350955973,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.301868342999967 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2624245105618133,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8106196630000113 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.24367360425114942,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.103850324999996 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4890568456813966,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0447520750000194 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.035841039382845,
            "unit": "iter/sec",
            "range": "stddev: 0.002031148276782495",
            "extra": "mean: 62.36030885714529 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.07723917381753,
            "unit": "iter/sec",
            "range": "stddev: 0.00012340590242919386",
            "extra": "mean: 7.29515713934079 msec\nrounds: 122"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.3989358566755032,
            "unit": "iter/sec",
            "range": "stddev: 0.10853610516108907",
            "extra": "mean: 294.2097297999908 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 72.91343108337531,
            "unit": "iter/sec",
            "range": "stddev: 0.000215081710955345",
            "extra": "mean: 13.714894295078727 msec\nrounds: 61"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.517474595220903,
            "unit": "iter/sec",
            "range": "stddev: 0.00023558592374502154",
            "extra": "mean: 133.02339599999868 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 14.86011830712426,
            "unit": "iter/sec",
            "range": "stddev: 0.001782041828469547",
            "extra": "mean: 67.29421524999424 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.055372710459583,
            "unit": "iter/sec",
            "range": "stddev: 0.0010988806639333926",
            "extra": "mean: 66.42147087499595 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0266415637637414,
            "unit": "iter/sec",
            "range": "stddev: 0.003980358715657859",
            "extra": "mean: 974.0497903999994 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.8875411388185555,
            "unit": "iter/sec",
            "range": "stddev: 0.0013229471172662623",
            "extra": "mean: 145.1896953999949 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 14.840437654895515,
            "unit": "iter/sec",
            "range": "stddev: 0.0008744593224752785",
            "extra": "mean: 67.38345749999652 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.158804943293553,
            "unit": "iter/sec",
            "range": "stddev: 0.0010861713757931155",
            "extra": "mean: 65.96826093750963 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1727693106304131,
            "unit": "iter/sec",
            "range": "stddev: 0.008447993872714466",
            "extra": "mean: 852.6826128000039 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d79c8ac22e90cf48cbca40f847baca4ebb13be8d",
          "message": "MAINT: update pip constraints and pre-commit config (#462)\n\n* FIX: restore release drafter template\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2022-10-04T11:46:26Z",
          "tree_id": "07e099ab4f715ed796fb9d5835e6e74da2e9bb24",
          "url": "https://github.com/ComPWA/tensorwaves/commit/d79c8ac22e90cf48cbca40f847baca4ebb13be8d"
        },
        "date": 1664884212250,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3000959731530008,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.3322673060000056 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2652020285636265,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.770710221999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.27309010294412034,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.661795096999981 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4632711264122263,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.158563189000006 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.271783126504005,
            "unit": "iter/sec",
            "range": "stddev: 0.0007837682427001841",
            "extra": "mean: 54.72919600000381 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.3119236850907,
            "unit": "iter/sec",
            "range": "stddev: 0.00015536585093129358",
            "extra": "mean: 7.282688736437677 msec\nrounds: 129"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5469223075642704,
            "unit": "iter/sec",
            "range": "stddev: 0.09752816905518696",
            "extra": "mean: 281.93456559997685 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 77.31193314968918,
            "unit": "iter/sec",
            "range": "stddev: 0.00021703027639080431",
            "extra": "mean: 12.934613833337064 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.334300862320331,
            "unit": "iter/sec",
            "range": "stddev: 0.0002094791066398302",
            "extra": "mean: 119.98606920000157 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.090760649441743,
            "unit": "iter/sec",
            "range": "stddev: 0.0007495483089262462",
            "extra": "mean: 66.2657120624992 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.28035874576775,
            "unit": "iter/sec",
            "range": "stddev: 0.0006841581628122471",
            "extra": "mean: 65.44348968750313 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.168306686797307,
            "unit": "iter/sec",
            "range": "stddev: 0.006271774763863959",
            "extra": "mean: 855.9396358000072 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.077255994528186,
            "unit": "iter/sec",
            "range": "stddev: 0.0023493213661997326",
            "extra": "mean: 141.29770080002118 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.258395166875589,
            "unit": "iter/sec",
            "range": "stddev: 0.003463874844199282",
            "extra": "mean: 65.53769181249791 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.525512756839248,
            "unit": "iter/sec",
            "range": "stddev: 0.0007301214121060667",
            "extra": "mean: 64.41011100000438 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.3368221007111136,
            "unit": "iter/sec",
            "range": "stddev: 0.00694971935607812",
            "extra": "mean: 748.0426897999791 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6d9d5bf84fe29357859882a06ced488bd00a961d",
          "message": "FIX: allow generating empty phase space samples (#465)\n\n* FIX: update PR labels\r\n* MAINT: add test for generating empty phase space sample",
          "timestamp": "2022-10-28T10:44:36Z",
          "tree_id": "b06fd6c82bb620b92074e6b624f2ff0c974fca98",
          "url": "https://github.com/ComPWA/tensorwaves/commit/6d9d5bf84fe29357859882a06ced488bd00a961d"
        },
        "date": 1666958051654,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2847733697659762,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.511564304000018 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2485793514814404,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.022860281999982 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.25989806714638863,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8476623200000404 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5164197788591434,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9364091790000089 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.477903980456684,
            "unit": "iter/sec",
            "range": "stddev: 0.0010385986462987527",
            "extra": "mean: 57.21509862499374 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 137.0719873074956,
            "unit": "iter/sec",
            "range": "stddev: 0.00013298234783215598",
            "extra": "mean: 7.295436650792006 msec\nrounds: 126"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.461652585300702,
            "unit": "iter/sec",
            "range": "stddev: 0.1033896997950881",
            "extra": "mean: 288.879364800016 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 74.0520147131848,
            "unit": "iter/sec",
            "range": "stddev: 0.00014694564782978273",
            "extra": "mean: 13.504021516135094 msec\nrounds: 62"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.67387301869163,
            "unit": "iter/sec",
            "range": "stddev: 0.0006280269812165523",
            "extra": "mean: 130.31229440000516 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 14.844933550211355,
            "unit": "iter/sec",
            "range": "stddev: 0.0006560186858771811",
            "extra": "mean: 67.36304993333988 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 14.850207617133394,
            "unit": "iter/sec",
            "range": "stddev: 0.0010478332361293712",
            "extra": "mean: 67.33912587499802 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0046886102773005,
            "unit": "iter/sec",
            "range": "stddev: 0.006314321172471365",
            "extra": "mean: 995.3332702000012 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.9437146889618555,
            "unit": "iter/sec",
            "range": "stddev: 0.0011528351298466532",
            "extra": "mean: 144.01513380001916 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 14.578353347580938,
            "unit": "iter/sec",
            "range": "stddev: 0.001320815151582077",
            "extra": "mean: 68.59485266667207 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 14.800795980517293,
            "unit": "iter/sec",
            "range": "stddev: 0.001808335983016077",
            "extra": "mean: 67.56393381250092 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.15920168498051,
            "unit": "iter/sec",
            "range": "stddev: 0.0016218092891360286",
            "extra": "mean: 862.6626522000038 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fda51fb00de4bbdbf18ff122cb77098066e6aa42",
          "message": "FIX: remove cast to numpy in `UnbinnedNLL` (#466)",
          "timestamp": "2022-10-31T12:34:20+01:00",
          "tree_id": "f09d45fc2dbdc617850a6e5b82f207b7a120d40c",
          "url": "https://github.com/ComPWA/tensorwaves/commit/fda51fb00de4bbdbf18ff122cb77098066e6aa42"
        },
        "date": 1667216277165,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.29353762870501304,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.4067182609999804 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2599444722526965,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8469754380000154 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.26826928203291034,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7275978540000096 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.45967767598102593,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.175437381999984 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.598526229021523,
            "unit": "iter/sec",
            "range": "stddev: 0.0006048354196694055",
            "extra": "mean: 56.82293999999339 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.27946134104465,
            "unit": "iter/sec",
            "range": "stddev: 0.0001589412431739244",
            "extra": "mean: 7.337862875004042 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5688274009874843,
            "unit": "iter/sec",
            "range": "stddev: 0.08697709274508597",
            "extra": "mean: 280.2040803999944 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 76.29954177224245,
            "unit": "iter/sec",
            "range": "stddev: 0.00022051320992743376",
            "extra": "mean: 13.106238606059323 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.415506777713592,
            "unit": "iter/sec",
            "range": "stddev: 0.000540382891663601",
            "extra": "mean: 134.85255019999158 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.146567516629853,
            "unit": "iter/sec",
            "range": "stddev: 0.0008016214824632912",
            "extra": "mean: 66.02155893749995 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.189838783929078,
            "unit": "iter/sec",
            "range": "stddev: 0.0006751346342276442",
            "extra": "mean: 65.83348343749407 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.184636896619053,
            "unit": "iter/sec",
            "range": "stddev: 0.007042165575500181",
            "extra": "mean: 844.1405149999923 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.132055135921564,
            "unit": "iter/sec",
            "range": "stddev: 0.0003793545035837634",
            "extra": "mean: 140.21203999999443 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.565234076473415,
            "unit": "iter/sec",
            "range": "stddev: 0.00024753812137375404",
            "extra": "mean: 64.24574118750215 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.582172520561064,
            "unit": "iter/sec",
            "range": "stddev: 0.0006698257709446476",
            "extra": "mean: 64.17590350000779 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.342455179887233,
            "unit": "iter/sec",
            "range": "stddev: 0.00863770331096374",
            "extra": "mean: 744.9038261999931 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "64215bfef61131e373596d37583870ef1e6fe60e",
          "message": "MAINT: activate and address pyright strict type checking (#467)\n\n* DX: ignore `flake8`'s `E302` error\r\n* DX: ignore `pylint`'s `line-too-long`\r\n* FIX: fix link to changelog on PyPI",
          "timestamp": "2022-10-31T12:57:42+01:00",
          "tree_id": "454169d76232cdcf4a56045a9d1a559435ebe305",
          "url": "https://github.com/ComPWA/tensorwaves/commit/64215bfef61131e373596d37583870ef1e6fe60e"
        },
        "date": 1667217693544,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.26986622546345446,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7055396549999955 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.24247056511783674,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.124211940999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.25676214240862,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8946551489999877 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5099722742557745,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9608909159999826 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.479566574591612,
            "unit": "iter/sec",
            "range": "stddev: 0.0007147462327366564",
            "extra": "mean: 60.68120757143041 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 132.68882669665382,
            "unit": "iter/sec",
            "range": "stddev: 0.00018339869475218992",
            "extra": "mean: 7.53642959166522 msec\nrounds: 120"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.999131698927433,
            "unit": "iter/sec",
            "range": "stddev: 0.0036293532799292795",
            "extra": "mean: 250.05428060001123 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 72.25028260668381,
            "unit": "iter/sec",
            "range": "stddev: 0.0002071731497494691",
            "extra": "mean: 13.840776311475507 msec\nrounds: 61"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.581440324761908,
            "unit": "iter/sec",
            "range": "stddev: 0.0005422170352207303",
            "extra": "mean: 131.90105800000538 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 14.494213995769881,
            "unit": "iter/sec",
            "range": "stddev: 0.0005311725666171696",
            "extra": "mean: 68.99304786667624 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 14.72646566122864,
            "unit": "iter/sec",
            "range": "stddev: 0.001115168013946781",
            "extra": "mean: 67.904955812498 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9799301876058363,
            "unit": "iter/sec",
            "range": "stddev: 0.018288267873415566",
            "extra": "mean: 1.0204808593999928 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.697430106323073,
            "unit": "iter/sec",
            "range": "stddev: 0.0003986050279290957",
            "extra": "mean: 149.31100199999037 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 13.687601519473965,
            "unit": "iter/sec",
            "range": "stddev: 0.005843973747879657",
            "extra": "mean: 73.05881885714273 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 12.892617292122793,
            "unit": "iter/sec",
            "range": "stddev: 0.008031471709156815",
            "extra": "mean: 77.56376981817229 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.138800219006231,
            "unit": "iter/sec",
            "range": "stddev: 0.006869924309112629",
            "extra": "mean: 878.117147599994 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "816e7e025a1e1c216ee1645a26c786a0d3acc027",
          "message": "MAINT: update pip constraints and pre-commit (#468)\n\n* DX: set wrapping column in git messages to 72\r\n* DX: shorten upgrade PR title\r\n* MAINT: limit `jupyterlab-server` for Python 3.7\r\n  https://github.com/ComPWA/tensorwaves/actions/runs/3366807875/jobs/5583673530#step:3:74\r\n* MAINT: change indent size of `.flake8` to 2 spaces\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-11-01T14:07:11Z",
          "tree_id": "43dfb3c4d5cdf034539f64f27c130ccba618b90f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/816e7e025a1e1c216ee1645a26c786a0d3acc027"
        },
        "date": 1667311847645,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2869358276902025,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.4850998150000123 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25274509288767844,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.956555549999962 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.26882573229157497,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.719881990000033 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5298829955448721,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.8872090790000016 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.784586579865945,
            "unit": "iter/sec",
            "range": "stddev: 0.0007856023648630605",
            "extra": "mean: 59.578470714289516 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 135.70868877053513,
            "unit": "iter/sec",
            "range": "stddev: 0.0001807289146409607",
            "extra": "mean: 7.368724943550694 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.4725455572090675,
            "unit": "iter/sec",
            "range": "stddev: 0.10388741558369798",
            "extra": "mean: 287.9731837999884 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 72.59309066319635,
            "unit": "iter/sec",
            "range": "stddev: 0.00024453408170274795",
            "extra": "mean: 13.775415688520694 msec\nrounds: 61"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.662847955864443,
            "unit": "iter/sec",
            "range": "stddev: 0.00029616634662999814",
            "extra": "mean: 130.4997835999984 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 14.233243609370039,
            "unit": "iter/sec",
            "range": "stddev: 0.0006823344400702785",
            "extra": "mean: 70.25805413333046 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 14.386387153923627,
            "unit": "iter/sec",
            "range": "stddev: 0.0015761469474583347",
            "extra": "mean: 69.51015493332307 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9856623776356813,
            "unit": "iter/sec",
            "range": "stddev: 0.004153242109235461",
            "extra": "mean: 1.0145461800000022 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.954787991522379,
            "unit": "iter/sec",
            "range": "stddev: 0.00016128175104299273",
            "extra": "mean: 143.78583520000348 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 14.579150288467002,
            "unit": "iter/sec",
            "range": "stddev: 0.0010963709689008404",
            "extra": "mean: 68.59110306250571 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 14.766319421699706,
            "unit": "iter/sec",
            "range": "stddev: 0.0012590858620861861",
            "extra": "mean: 67.72168280001173 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1097542737475035,
            "unit": "iter/sec",
            "range": "stddev: 0.02209575250646016",
            "extra": "mean: 901.1003819999928 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "340004bdeea7b0c9f8f310934fa1d2b9b7662b80",
          "message": "DX: switch to new VSCode Python extensions (#469)\n\n* DX: activate and configure VSCode flake8 extension\r\n* DX: activate and configure VSCode isort extension\r\n* DX: activate and configure VSCode pylint extension\r\n* DX: extend pyright file exclusions\r\n* MAINT: limit scope of pyright ignore statements",
          "timestamp": "2022-11-06T11:47:15Z",
          "tree_id": "b0de5274acf7c5e86fe91df71ff1349ec40009c9",
          "url": "https://github.com/ComPWA/tensorwaves/commit/340004bdeea7b0c9f8f310934fa1d2b9b7662b80"
        },
        "date": 1667735443112,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.30376612731885094,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.2920062839999957 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25226954688876313,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9640139379999937 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.26025179478258537,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8424326749999977 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.6018474644395564,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.6615505740000174 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.65580137869722,
            "unit": "iter/sec",
            "range": "stddev: 0.001137396478307774",
            "extra": "mean: 53.602629000000235 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 155.5496209779701,
            "unit": "iter/sec",
            "range": "stddev: 0.00018977741299364077",
            "extra": "mean: 6.428816693430748 msec\nrounds: 137"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.711868009632109,
            "unit": "iter/sec",
            "range": "stddev: 0.1284449715527235",
            "extra": "mean: 269.4061311999917 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 82.4776027945575,
            "unit": "iter/sec",
            "range": "stddev: 0.0003157392821871695",
            "extra": "mean: 12.124503696971994 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 8.457521118405928,
            "unit": "iter/sec",
            "range": "stddev: 0.0018551025809488565",
            "extra": "mean: 118.23795483332826 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.82511760648871,
            "unit": "iter/sec",
            "range": "stddev: 0.0010885781856313581",
            "extra": "mean: 63.190683624997135 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 16.001384055716265,
            "unit": "iter/sec",
            "range": "stddev: 0.0030358896929989273",
            "extra": "mean: 62.49459399999617 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0892754406793632,
            "unit": "iter/sec",
            "range": "stddev: 0.0038465420320888044",
            "extra": "mean: 918.0414453999958 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.591559418188844,
            "unit": "iter/sec",
            "range": "stddev: 0.0027291249485609003",
            "extra": "mean: 131.72524180000096 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.902026880893763,
            "unit": "iter/sec",
            "range": "stddev: 0.0012112502529063058",
            "extra": "mean: 62.88506537499927 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 16.10235228355117,
            "unit": "iter/sec",
            "range": "stddev: 0.003068373885140414",
            "extra": "mean: 62.10272775000192 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2680416834405475,
            "unit": "iter/sec",
            "range": "stddev: 0.01604500724642107",
            "extra": "mean: 788.6176085999978 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "00ce851a5ab257b11366e4933ad2c65640a149f0",
          "message": "FEAT: implement `ChainedDataTransformer` (#470)",
          "timestamp": "2022-11-06T12:12:10Z",
          "tree_id": "0f5678701e817e65c66e3e9babb8ed316952875f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/00ce851a5ab257b11366e4933ad2c65640a149f0"
        },
        "date": 1667736958060,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.279890039096535,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5728316850000397 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2434268277104581,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.108010647000015 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.26193436931318753,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8177502350000054 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5191923334822227,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.926068502000021 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.989680424719438,
            "unit": "iter/sec",
            "range": "stddev: 0.0007266777332260417",
            "extra": "mean: 62.5403368571418 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 134.64520822703517,
            "unit": "iter/sec",
            "range": "stddev: 0.0002343206992094672",
            "extra": "mean: 7.426926016660218 msec\nrounds: 120"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.3387060971522495,
            "unit": "iter/sec",
            "range": "stddev: 0.12376136986028316",
            "extra": "mean: 299.5172294000213 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 74.65774537801735,
            "unit": "iter/sec",
            "range": "stddev: 0.0001629881665908806",
            "extra": "mean: 13.394457533329765 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.636684444313053,
            "unit": "iter/sec",
            "range": "stddev: 0.0006335074044242587",
            "extra": "mean: 130.94687980000117 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 14.938783182030734,
            "unit": "iter/sec",
            "range": "stddev: 0.0004150822106474198",
            "extra": "mean: 66.93985633333645 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.097571927331611,
            "unit": "iter/sec",
            "range": "stddev: 0.0006948347500372382",
            "extra": "mean: 66.23581625000696 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0278415280896698,
            "unit": "iter/sec",
            "range": "stddev: 0.004499096569799096",
            "extra": "mean: 972.9126257999951 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.179782487255634,
            "unit": "iter/sec",
            "range": "stddev: 0.0005937943443132497",
            "extra": "mean: 139.27998539998043 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.047745276893767,
            "unit": "iter/sec",
            "range": "stddev: 0.0013617645175965144",
            "extra": "mean: 66.45513873334417 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.01873647920139,
            "unit": "iter/sec",
            "range": "stddev: 0.000957468293360042",
            "extra": "mean: 66.58349731249658 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1876124502427845,
            "unit": "iter/sec",
            "range": "stddev: 0.002911402484806481",
            "extra": "mean: 842.0255275999921 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2392c49487a5feb188b4fc4eee4de7358d86cc50",
          "message": "DX: implement GitHub Actions caching (#471)\n\n* FIX: make `Estimator` call abstract\r\n  https://github.com/ComPWA/tensorwaves/actions/runs/3504245141/jobs/5869810766#step:7:41\r\n* MAINT: remove redundant comments\r\n* MAINT: ignore mypy issues in `_relink_references.py`\r\n* MAINT: upgrade `actions/checkout` to v3\r\n* MAINT: autoupdate pre-commit config\r\n* MAINT: split lines in `strategy.matrix` definition\r\n* MAINT: set `PYTHONHASHSEED=0` in workflows\r\n* MAINT: define all jobs in pytest workflow with matrix\r\n* MAINT: remove deprecated `set-output`\r\n  https://github.blog/changelog/2022-10-11-github-actions-deprecating-save-state-and-set-output-commands/\r\n* MAINT: update pip constraints and pre-commit\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-11-25T14:49:11+01:00",
          "tree_id": "8d47cc7ee5ddcd72d697061e3eda2cd1f69ecd25",
          "url": "https://github.com/ComPWA/tensorwaves/commit/2392c49487a5feb188b4fc4eee4de7358d86cc50"
        },
        "date": 1669384409476,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.23099783521810877,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.3290448980000065 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2011153930491113,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.972269824000023 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20482888676423944,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.88212388300002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4031633675004653,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4803840840000078 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 13.025355376033636,
            "unit": "iter/sec",
            "range": "stddev: 0.0031391885457610225",
            "extra": "mean: 76.77333716667552 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 106.05621855673543,
            "unit": "iter/sec",
            "range": "stddev: 0.0008889354528878315",
            "extra": "mean: 9.428961484847244 msec\nrounds: 99"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.2126223386826833,
            "unit": "iter/sec",
            "range": "stddev: 0.006781500124500531",
            "extra": "mean: 311.2721927999928 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 58.61832989680598,
            "unit": "iter/sec",
            "range": "stddev: 0.0011064309790896816",
            "extra": "mean: 17.05951025490558 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.710776712849663,
            "unit": "iter/sec",
            "range": "stddev: 0.004752621729671683",
            "extra": "mean: 175.1075291999996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 11.160388057245902,
            "unit": "iter/sec",
            "range": "stddev: 0.002727329949169235",
            "extra": "mean: 89.60261909089695 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 12.459178643266371,
            "unit": "iter/sec",
            "range": "stddev: 0.0035353243863378726",
            "extra": "mean: 80.26211266666887 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.902404606638933,
            "unit": "iter/sec",
            "range": "stddev: 0.0034538987731324708",
            "extra": "mean: 1.108150371399995 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.141132950474439,
            "unit": "iter/sec",
            "range": "stddev: 0.0011654551003808938",
            "extra": "mean: 162.83640299999433 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 12.728815376025459,
            "unit": "iter/sec",
            "range": "stddev: 0.0006764346486769747",
            "extra": "mean: 78.5619062307625 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 12.631570768190372,
            "unit": "iter/sec",
            "range": "stddev: 0.0013359981656758599",
            "extra": "mean: 79.16671792856229 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.0412082634489337,
            "unit": "iter/sec",
            "range": "stddev: 0.007747902834702514",
            "extra": "mean: 960.4226503999939 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "39f64aa0f6ffa729cceada748fec1d64d4f2c32a",
          "message": "MAINT: update pip constraints and pre-commit (#472)\n\n* DX: add link to diff with previous release\r\n* FIX: exclude IPython v8.7.0\r\n* MAINT: address `pydocstyle` issues\r\n* MAINT: limit `tox` version for Python 3.7\r\n  https://github.com/ComPWA/tensorwaves/actions/runs/3658736023/jobs/6183905960#step:3:88\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2022-12-09T19:10:52Z",
          "tree_id": "4bdd81aa83780246fb980b76afb09b2d2b532c4b",
          "url": "https://github.com/ComPWA/tensorwaves/commit/39f64aa0f6ffa729cceada748fec1d64d4f2c32a"
        },
        "date": 1670613264872,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2842602320899013,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5179032699999198 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2611650798867942,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.828995823000014 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.27108869164618266,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.688829637000026 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.46993609778395795,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.1279488950000314 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 17.26221897539145,
            "unit": "iter/sec",
            "range": "stddev: 0.00043962704243686603",
            "extra": "mean: 57.92997999999727 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.56088584462512,
            "unit": "iter/sec",
            "range": "stddev: 0.00012246170958942667",
            "extra": "mean: 7.2170439291312505 msec\nrounds: 127"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.5735154227551718,
            "unit": "iter/sec",
            "range": "stddev: 0.08763324143177244",
            "extra": "mean: 279.836486399995 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 77.51829109263392,
            "unit": "iter/sec",
            "range": "stddev: 0.00023427274604835699",
            "extra": "mean: 12.900181181819473 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.504888267695422,
            "unit": "iter/sec",
            "range": "stddev: 0.0006273367928663253",
            "extra": "mean: 133.2464874000152 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.577369257526648,
            "unit": "iter/sec",
            "range": "stddev: 0.0003098938311634576",
            "extra": "mean: 64.19569206249776 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.827916329543683,
            "unit": "iter/sec",
            "range": "stddev: 0.00030201952358828286",
            "extra": "mean: 63.17951012499634 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2074461238668566,
            "unit": "iter/sec",
            "range": "stddev: 0.0014025472378239121",
            "extra": "mean: 828.194302199995 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.140528065812756,
            "unit": "iter/sec",
            "range": "stddev: 0.00032370929592397444",
            "extra": "mean: 140.04566480002723 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.7821202544662,
            "unit": "iter/sec",
            "range": "stddev: 0.00020218945954964524",
            "extra": "mean: 63.36284250001256 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.572287578436766,
            "unit": "iter/sec",
            "range": "stddev: 0.0004307496998021833",
            "extra": "mean: 64.21664093750223 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.3859228864838895,
            "unit": "iter/sec",
            "range": "stddev: 0.0018445767761031714",
            "extra": "mean: 721.5408662000073 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "678b5a17396e6eb7a8da2881cfef0005d5c9bd6d",
          "message": "FEAT: add `migrad_args` option to `Minuit2` optimizer (#476)",
          "timestamp": "2023-01-23T13:31:47+01:00",
          "tree_id": "71a8689f8dea6e0b58b1226c6d89d9784cf58985",
          "url": "https://github.com/ComPWA/tensorwaves/commit/678b5a17396e6eb7a8da2881cfef0005d5c9bd6d"
        },
        "date": 1674477362789,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2677596105583393,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7346932120000247 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.23015408569311932,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.344915264000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21645327945152584,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.619934623000006 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.43491492286263606,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.299300270999993 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.277943587765225,
            "unit": "iter/sec",
            "range": "stddev: 0.003317053520622412",
            "extra": "mean: 70.03809714284769 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 117.20698224639403,
            "unit": "iter/sec",
            "range": "stddev: 0.0009342098845075598",
            "extra": "mean: 8.531914915254683 msec\nrounds: 118"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.9717004031551646,
            "unit": "iter/sec",
            "range": "stddev: 0.12088968729345637",
            "extra": "mean: 336.50767719998385 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 61.96556597046674,
            "unit": "iter/sec",
            "range": "stddev: 0.0017434360051467822",
            "extra": "mean: 16.13799509999808 msec\nrounds: 50"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.484371588713809,
            "unit": "iter/sec",
            "range": "stddev: 0.0013870204226948929",
            "extra": "mean: 154.21694860000343 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 12.314910209188763,
            "unit": "iter/sec",
            "range": "stddev: 0.002510853376613232",
            "extra": "mean: 81.20237849999512 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 12.353593373533428,
            "unit": "iter/sec",
            "range": "stddev: 0.003443964767125253",
            "extra": "mean: 80.94810714285116 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9477308747400028,
            "unit": "iter/sec",
            "range": "stddev: 0.022306893763729938",
            "extra": "mean: 1.055151865000005 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.661872365834834,
            "unit": "iter/sec",
            "range": "stddev: 0.0020410353053450106",
            "extra": "mean: 150.10794940000096 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 13.551123977945169,
            "unit": "iter/sec",
            "range": "stddev: 0.0015793542320008862",
            "extra": "mean: 73.79461671426871 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 13.599011848656057,
            "unit": "iter/sec",
            "range": "stddev: 0.002304816423579897",
            "extra": "mean: 73.53475466666546 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1006547193307232,
            "unit": "iter/sec",
            "range": "stddev: 0.008822701619250679",
            "extra": "mean: 908.5501406000162 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9a607df7822e5f3d2e5fffc8bb3fd426cee6c233",
          "message": "DX: outsource GitHub workflows to ComPWA/actions (#477)\n\n* DX: merge CD workflows\r\n\r\n* DX: rename \"✨ Feature\" issue label\r\n\r\n* DX: remove VSCode GitHub Actions extenstion\r\n\r\n* DX: run `editorconfig-checker` on pre-commit.ci\r\n\r\n* DX: run `pip install` with ComPWA/actions in benchmarks\r\n\r\n* FIX: update `isort` pre-commit hook\r\n  https://results.pre-commit.ci/run/github/244342170/1675085162.1LHV7yF-S7WtIRkZueudPw\r\n\r\n* MAINT: commit empty `tests/output/` directory\r\n\r\n* MAINT: move nbQA hooks to top of pre-commit config\r\n\r\n* MAINT: upgrade ComPWA/repo-maintenance hook",
          "timestamp": "2023-01-30T14:35:10+01:00",
          "tree_id": "fe6de722f1291e554746d42dfdf17637d0cbd5fa",
          "url": "https://github.com/ComPWA/tensorwaves/commit/9a607df7822e5f3d2e5fffc8bb3fd426cee6c233"
        },
        "date": 1675085992035,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.21928158636874284,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.560346432000017 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19366188079285382,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.163638791000011 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.1857584800582829,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.383334315000013 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.40940283616368917,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4425820040000303 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 10.66683726020798,
            "unit": "iter/sec",
            "range": "stddev: 0.011545208053364937",
            "extra": "mean: 93.74850066668235 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 89.16107787053565,
            "unit": "iter/sec",
            "range": "stddev: 0.0007717089018265621",
            "extra": "mean: 11.215656246911099 msec\nrounds: 81"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.4306566945054726,
            "unit": "iter/sec",
            "range": "stddev: 0.11620463875404424",
            "extra": "mean: 411.411451999993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 49.20712206349,
            "unit": "iter/sec",
            "range": "stddev: 0.0017862395571341806",
            "extra": "mean: 20.322261454545945 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.83301754267804,
            "unit": "iter/sec",
            "range": "stddev: 0.013022041580932517",
            "extra": "mean: 206.91007040001068 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.827723612234042,
            "unit": "iter/sec",
            "range": "stddev: 0.009770438316327497",
            "extra": "mean: 207.13696150000752 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.045794205293086,
            "unit": "iter/sec",
            "range": "stddev: 0.0023116973750613628",
            "extra": "mean: 198.18485640000745 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.5840296980484471,
            "unit": "iter/sec",
            "range": "stddev: 0.02872945638571436",
            "extra": "mean: 1.712241694799991 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.905316593149751,
            "unit": "iter/sec",
            "range": "stddev: 0.00709410778575071",
            "extra": "mean: 203.860440200026 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.0430707449404375,
            "unit": "iter/sec",
            "range": "stddev: 0.005209068336133608",
            "extra": "mean: 198.2918841666598 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.733095185335509,
            "unit": "iter/sec",
            "range": "stddev: 0.02357721088262741",
            "extra": "mean: 211.27823566664952 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.6765989666485865,
            "unit": "iter/sec",
            "range": "stddev: 0.04018568600282367",
            "extra": "mean: 1.4779803831999971 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f157490fc78ebd9bbb6ce4f533e46e4b1d9124b9",
          "message": "MAINT: update pip constraints and pre-commit (#474)\n\n* DX: remove `docformatter` pre-commit hook\r\n\r\n* MAINT: ignore new `mypy` errors\r\n\r\n* MAINT: render JAX output in doctest with `tolist()`\r\n  This is to support both JAX v0.3 and v0.4\r\n\r\n* MAINT: update test values to JAX v0.4.x\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-01-30T14:06:15Z",
          "tree_id": "caf67c382930290e1f7676d10147d787c955eb4b",
          "url": "https://github.com/ComPWA/tensorwaves/commit/f157490fc78ebd9bbb6ce4f533e46e4b1d9124b9"
        },
        "date": 1675087790383,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.29466047431119,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.393736476999976 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2648449545183708,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.775794036999997 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.25972814473781397,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.850179582999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5036140345309872,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9856476020000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.04299036920652,
            "unit": "iter/sec",
            "range": "stddev: 0.007522099198030119",
            "extra": "mean: 62.332518874999465 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.47352047515966,
            "unit": "iter/sec",
            "range": "stddev: 0.00016363878557873202",
            "extra": "mean: 7.327428768000573 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.177096915955995,
            "unit": "iter/sec",
            "range": "stddev: 0.0006789481748680808",
            "extra": "mean: 239.40071779999244 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 79.0243670241254,
            "unit": "iter/sec",
            "range": "stddev: 0.00024368018604328334",
            "extra": "mean: 12.654324705881029 msec\nrounds: 68"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.756780693007865,
            "unit": "iter/sec",
            "range": "stddev: 0.0002155390638917205",
            "extra": "mean: 128.91946280000184 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.285929190594002,
            "unit": "iter/sec",
            "range": "stddev: 0.00036491750069656974",
            "extra": "mean: 65.41964099999475 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.329256810016942,
            "unit": "iter/sec",
            "range": "stddev: 0.0006868876953366838",
            "extra": "mean: 65.23473462500462 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2671394328241739,
            "unit": "iter/sec",
            "range": "stddev: 0.00900260227612667",
            "extra": "mean: 789.1791337999962 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.912081206624013,
            "unit": "iter/sec",
            "range": "stddev: 0.0018516053121277211",
            "extra": "mean: 144.6742262000157 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 14.517232735317432,
            "unit": "iter/sec",
            "range": "stddev: 0.000322612163867952",
            "extra": "mean: 68.88365146666047 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 16.241743206503152,
            "unit": "iter/sec",
            "range": "stddev: 0.0007674291560760532",
            "extra": "mean: 61.56974576470355 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4397090370150276,
            "unit": "iter/sec",
            "range": "stddev: 0.0011008837214658222",
            "extra": "mean: 694.5847905999926 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6548b77cb1201a6c857263cfe518b28d4d7906d1",
          "message": "FIX: allow higher Python versions (#478)\n\n* MAINT: clean up version constraints\r\n\r\n* MAINT: do not install `flake8` for Python 3.7\r\n\r\n* MAINT: update pip constraints and pre-commit\r\n\r\n---------\r\n\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2023-02-07T19:41:44+01:00",
          "tree_id": "49205d0a6702f0c23755545933f1db98f0d12bd4",
          "url": "https://github.com/ComPWA/tensorwaves/commit/6548b77cb1201a6c857263cfe518b28d4d7906d1"
        },
        "date": 1675795562521,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.23549183992044528,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.246431640000026 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2104155652178975,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.752500124999983 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.20666485506596993,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.838752092999982 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.45549299073378763,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.195423465000033 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.445465509563435,
            "unit": "iter/sec",
            "range": "stddev: 0.0003125776042042639",
            "extra": "mean: 69.22587571428299 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 113.19950770770059,
            "unit": "iter/sec",
            "range": "stddev: 0.00024815318624258806",
            "extra": "mean: 8.833960679247488 msec\nrounds: 106"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.470688606873587,
            "unit": "iter/sec",
            "range": "stddev: 0.0012284623266814877",
            "extra": "mean: 288.127260400006 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 46.59206472333085,
            "unit": "iter/sec",
            "range": "stddev: 0.03976715866476754",
            "extra": "mean: 21.46288227272428 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.021420538802074,
            "unit": "iter/sec",
            "range": "stddev: 0.0005475963880850663",
            "extra": "mean: 166.07376839999688 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 12.3433223520975,
            "unit": "iter/sec",
            "range": "stddev: 0.0011454998940501596",
            "extra": "mean: 81.01546499999411 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 12.341763766530157,
            "unit": "iter/sec",
            "range": "stddev: 0.0007126510959196974",
            "extra": "mean: 81.0256960769187 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8884898924820339,
            "unit": "iter/sec",
            "range": "stddev: 0.0025245860516104834",
            "extra": "mean: 1.1255052065999962 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.8035498357385515,
            "unit": "iter/sec",
            "range": "stddev: 0.00008166132506825293",
            "extra": "mean: 172.308333400008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 12.174570282970809,
            "unit": "iter/sec",
            "range": "stddev: 0.0006790048102777135",
            "extra": "mean: 82.13842269231883 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 11.858487902223743,
            "unit": "iter/sec",
            "range": "stddev: 0.006388092617081355",
            "extra": "mean: 84.3277834615387 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9955879018790341,
            "unit": "iter/sec",
            "range": "stddev: 0.012195872683838226",
            "extra": "mean: 1.0044316510000157 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "43057f2226d93769f313df2c9de774b58f517221",
          "message": "DX: colorize `sphinx-build` output (#481)\n\n* DX: pass all environment variables in `tox` config\r\n\r\n* MAINT: remove quotations from environment values\r\n\r\n* MAINT: sort config keys in `tox.ini`",
          "timestamp": "2023-03-07T18:18:23+01:00",
          "tree_id": "6543466466803ef3bb319099f20ba019ae4a4847",
          "url": "https://github.com/ComPWA/tensorwaves/commit/43057f2226d93769f313df2c9de774b58f517221"
        },
        "date": 1678209704965,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3022238656269501,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.308805537000012 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26730529315580986,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.74104077100003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2763439016588168,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6186794570000416 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5120912756733562,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9527768730000048 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.203935176979734,
            "unit": "iter/sec",
            "range": "stddev: 0.0025395363779668325",
            "extra": "mean: 54.933177374998365 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 139.53621326348372,
            "unit": "iter/sec",
            "range": "stddev: 0.0001380083200029702",
            "extra": "mean: 7.166598380534507 msec\nrounds: 113"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.277427401960434,
            "unit": "iter/sec",
            "range": "stddev: 0.001536962035933799",
            "extra": "mean: 233.7853821999829 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 80.70900743540506,
            "unit": "iter/sec",
            "range": "stddev: 0.0001993413413595252",
            "extra": "mean: 12.390190782613002 msec\nrounds: 69"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.87389527123447,
            "unit": "iter/sec",
            "range": "stddev: 0.00023895828091148922",
            "extra": "mean: 127.00194319999127 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.458850359657085,
            "unit": "iter/sec",
            "range": "stddev: 0.0007923631437618097",
            "extra": "mean: 64.68786337499566 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.316355454534142,
            "unit": "iter/sec",
            "range": "stddev: 0.0010475206268591925",
            "extra": "mean: 65.28968349999786 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2468650579944458,
            "unit": "iter/sec",
            "range": "stddev: 0.003583336248928785",
            "extra": "mean: 802.0114073999935 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 7.1359913914587745,
            "unit": "iter/sec",
            "range": "stddev: 0.0001692230734809534",
            "extra": "mean: 140.13469820001774 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.943894344603947,
            "unit": "iter/sec",
            "range": "stddev: 0.00028675032105700775",
            "extra": "mean: 62.71993393749753 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.713931229043478,
            "unit": "iter/sec",
            "range": "stddev: 0.0004996213369030788",
            "extra": "mean: 63.63779918749657 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4363398339875475,
            "unit": "iter/sec",
            "range": "stddev: 0.0009718309124852614",
            "extra": "mean: 696.214068800009 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7def59eeaa8fb10b6f3000fbb6fbe2327ffd0eee",
          "message": "MAINT: update pip constraints and pre-commit (#480)\n\n* DOC: add \"Last updated\" timestamp\r\n\r\n* DX: update and apply taplo config\r\n\r\n* FIX: add switch for JAX function name test\r\n  https://github.com/ComPWA/tensorwaves/actions/runs/4356965655/jobs/7615683115\r\n\r\n* MAINT: do not install TF v2.12\r\n  Benchmarks fail since this version\r\n\r\n* MAINT: remove redundant `None` comparison\r\n\r\n* MAINT: resolve `ypy-websocket` dependency conflict\r\n  https://github.com/ComPWA/tensorwaves/actions/runs/4350354717/jobs/7600982077#step:3:78\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-03-28T00:38:18+02:00",
          "tree_id": "7ca4a29fef06708c735b7627a31933bbedea6023",
          "url": "https://github.com/ComPWA/tensorwaves/commit/7def59eeaa8fb10b6f3000fbb6fbe2327ffd0eee"
        },
        "date": 1679956928884,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3104842171200863,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.2207756299999915 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2648991796583137,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7750211279999917 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.25132827382274325,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.978859937999971 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5033414252564602,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9867230269999823 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.781120223334458,
            "unit": "iter/sec",
            "range": "stddev: 0.0003592940568512422",
            "extra": "mean: 50.55325424999779 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 132.54760947862192,
            "unit": "iter/sec",
            "range": "stddev: 0.00019593963828734215",
            "extra": "mean: 7.544458960320111 msec\nrounds: 126"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.149746631534176,
            "unit": "iter/sec",
            "range": "stddev: 0.00033535619886328183",
            "extra": "mean: 240.97856780000484 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 77.49690233599901,
            "unit": "iter/sec",
            "range": "stddev: 0.00024604121887761705",
            "extra": "mean: 12.903741567170718 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.7757055116633,
            "unit": "iter/sec",
            "range": "stddev: 0.002114934439267746",
            "extra": "mean: 147.58610720000434 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.144272790266077,
            "unit": "iter/sec",
            "range": "stddev: 0.0004671458179122915",
            "extra": "mean: 66.03156281249412 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.366025581640104,
            "unit": "iter/sec",
            "range": "stddev: 0.0005291554411032785",
            "extra": "mean: 65.07863693750693 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2125774898918198,
            "unit": "iter/sec",
            "range": "stddev: 0.000748257779387415",
            "extra": "mean: 824.6895627999947 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.797456407581836,
            "unit": "iter/sec",
            "range": "stddev: 0.000821714996447439",
            "extra": "mean: 147.1138526000118 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.80775701397103,
            "unit": "iter/sec",
            "range": "stddev: 0.0006943949345289191",
            "extra": "mean: 63.260081687502634 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.796321355284713,
            "unit": "iter/sec",
            "range": "stddev: 0.0008527605142003163",
            "extra": "mean: 63.305878470587494 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.3922464028557975,
            "unit": "iter/sec",
            "range": "stddev: 0.0017033652364822542",
            "extra": "mean: 718.2636622000132 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "04bb3e6962fc67f186397f2c9205bdcc7aecac1a",
          "message": "DOC: improve documentation sidebar (#483)\n\n* DOC: add sidebar icons\r\n\r\n* DOC: merge ComPWA project links\r\n\r\n* DOC: remove API from main page (only sidebar)\r\n\r\n* FIX: add back package title to sidebar",
          "timestamp": "2023-03-31T13:04:29Z",
          "tree_id": "5505c92c128bd4aee3dffd323b002a9b8d761d99",
          "url": "https://github.com/ComPWA/tensorwaves/commit/04bb3e6962fc67f186397f2c9205bdcc7aecac1a"
        },
        "date": 1680268137960,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.224468790729042,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.4549622989999875 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.19850641226588894,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.037620641999979 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.19522081378222594,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.122404627999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.41919678668693666,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.3855144690000145 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 14.574262494537956,
            "unit": "iter/sec",
            "range": "stddev: 0.00454587426802531",
            "extra": "mean: 68.61410657141472 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 92.91865083004232,
            "unit": "iter/sec",
            "range": "stddev: 0.0009293916723763008",
            "extra": "mean: 10.76210202222051 msec\nrounds: 90"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.049189809929406,
            "unit": "iter/sec",
            "range": "stddev: 0.0069518142568605334",
            "extra": "mean: 327.95596939999996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 53.27163983080643,
            "unit": "iter/sec",
            "range": "stddev: 0.0012567965771311026",
            "extra": "mean: 18.77171423999812 msec\nrounds: 50"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.025348066713451,
            "unit": "iter/sec",
            "range": "stddev: 0.0032347660684320123",
            "extra": "mean: 198.9911915999869 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 5.2023398584793785,
            "unit": "iter/sec",
            "range": "stddev: 0.004747037208605184",
            "extra": "mean: 192.22119799998913 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 5.136988320100592,
            "unit": "iter/sec",
            "range": "stddev: 0.008340535394334334",
            "extra": "mean: 194.66659016667145 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.6933140112597733,
            "unit": "iter/sec",
            "range": "stddev: 0.02393135672365629",
            "extra": "mean: 1.4423478881999927 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.097809499194206,
            "unit": "iter/sec",
            "range": "stddev: 0.005036502450920194",
            "extra": "mean: 196.1626851999995 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 5.33310429902112,
            "unit": "iter/sec",
            "range": "stddev: 0.0026325513166226486",
            "extra": "mean: 187.5080523333376 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 5.415074811790145,
            "unit": "iter/sec",
            "range": "stddev: 0.004469873555487061",
            "extra": "mean: 184.6696554999975 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.8030855373845627,
            "unit": "iter/sec",
            "range": "stddev: 0.014780981292527345",
            "extra": "mean: 1.245197371200004 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ccad808e342b1f9117485bf7a6718155989ee08b",
          "message": "FIX: remove simplification step `Abs` of mass (#482)\n\n* DX: assert that simplification reduces operations\r\n\r\n* DX: fold step sections in analysis notebook\r\n\r\n* FIX: hide TensorFlow GPU warnings\r\n\r\n* FIX: remove redundant simplification step\r\n\r\n* MAINT: remove empty `metadata.tags`",
          "timestamp": "2023-03-31T13:41:32Z",
          "tree_id": "4c327ce6c827f414d6af762b1a5aa7e96b1c9f41",
          "url": "https://github.com/ComPWA/tensorwaves/commit/ccad808e342b1f9117485bf7a6718155989ee08b"
        },
        "date": 1680270307674,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.29211341874622987,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.4233278439999992 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.24481954624098035,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.084641178999988 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.24587159705964645,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.067163560000012 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.56222609360241,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.7786438789999863 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.502985859593725,
            "unit": "iter/sec",
            "range": "stddev: 0.000978444818527663",
            "extra": "mean: 51.274200125007496 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 136.2528827829734,
            "unit": "iter/sec",
            "range": "stddev: 0.00013467203132005496",
            "extra": "mean: 7.3392942562017005 msec\nrounds: 121"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.110214777941502,
            "unit": "iter/sec",
            "range": "stddev: 0.0003909887356961028",
            "extra": "mean: 243.29628840000055 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 76.46090905316572,
            "unit": "iter/sec",
            "range": "stddev: 0.0002583668086902512",
            "extra": "mean: 13.078578483871647 msec\nrounds: 62"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.206117683762495,
            "unit": "iter/sec",
            "range": "stddev: 0.0005194160125392997",
            "extra": "mean: 138.7709781999945 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.606371068064187,
            "unit": "iter/sec",
            "range": "stddev: 0.0005450933559643448",
            "extra": "mean: 64.07639518749697 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.815460194163363,
            "unit": "iter/sec",
            "range": "stddev: 0.0008313947562123424",
            "extra": "mean: 63.22926982352663 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.070627994815496,
            "unit": "iter/sec",
            "range": "stddev: 0.0018941352956927562",
            "extra": "mean: 934.0312460000007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.659817106930335,
            "unit": "iter/sec",
            "range": "stddev: 0.00028165318863454454",
            "extra": "mean: 150.1542736000033 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.664219074349003,
            "unit": "iter/sec",
            "range": "stddev: 0.00027789814382264244",
            "extra": "mean: 63.83976087499654 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.85032964841992,
            "unit": "iter/sec",
            "range": "stddev: 0.00029659481286979563",
            "extra": "mean: 63.09017049999888 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2350344554741475,
            "unit": "iter/sec",
            "range": "stddev: 0.0044817767382275735",
            "extra": "mean: 809.6940093999933 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "66853113+pre-commit-ci[bot]@users.noreply.github.com",
            "name": "pre-commit-ci[bot]",
            "username": "pre-commit-ci[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2a5d691a4a909dcf6f454c3dc356f5e835822f6b",
          "message": "MAINT: update pip constraints and pre-commit (#485)\n\n* DX: ignore indent in Markdown files\r\n* DX: pin CI workflow for GitHub Actions extension\r\n* FIX: adapt syntax to `pandas` v2\r\n* FIX: hide TF warnings in basics notebook\r\n* MAINT: hide matplotlib warnings\r\n* MAINT: resolve `ypy-websocket` dependency conflict\r\n\r\n---------\r\n\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: Remco de Boer <29308176+redeboer@users.noreply.github.com>",
          "timestamp": "2023-04-13T17:55:09+02:00",
          "tree_id": "3cefbe806024a622bf784463940a3dd8cd9d32f7",
          "url": "https://github.com/ComPWA/tensorwaves/commit/2a5d691a4a909dcf6f454c3dc356f5e835822f6b"
        },
        "date": 1681401610223,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2076829612034077,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.815031498999986 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.17764262404814912,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.629279601999997 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.18003049604624027,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.5546144790000085 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4141727873535345,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4144512400000053 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 13.951636113947337,
            "unit": "iter/sec",
            "range": "stddev: 0.0007710651818173354",
            "extra": "mean: 71.67618133333538 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 85.16549897239831,
            "unit": "iter/sec",
            "range": "stddev: 0.0012144833582203548",
            "extra": "mean: 11.741843963411695 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.5347813058621034,
            "unit": "iter/sec",
            "range": "stddev: 0.13722234078452294",
            "extra": "mean: 394.51135200000635 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 50.28888416711539,
            "unit": "iter/sec",
            "range": "stddev: 0.0005394076641472074",
            "extra": "mean: 19.885110130439404 msec\nrounds: 46"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 4.903967170364489,
            "unit": "iter/sec",
            "range": "stddev: 0.00140751295845495",
            "extra": "mean: 203.91653639999276 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 4.828133193764863,
            "unit": "iter/sec",
            "range": "stddev: 0.005370099256554432",
            "extra": "mean: 207.11938959998406 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 4.881038347427063,
            "unit": "iter/sec",
            "range": "stddev: 0.0027719262424657034",
            "extra": "mean: 204.87444039998763 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.6456401624641704,
            "unit": "iter/sec",
            "range": "stddev: 0.019702689545345212",
            "extra": "mean: 1.5488503629999855 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 4.647769901375962,
            "unit": "iter/sec",
            "range": "stddev: 0.0028017404319202084",
            "extra": "mean: 215.15695079998522 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 4.864071228805667,
            "unit": "iter/sec",
            "range": "stddev: 0.000964742552151812",
            "extra": "mean: 205.5890945999863 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 4.889489934111145,
            "unit": "iter/sec",
            "range": "stddev: 0.002456600679143683",
            "extra": "mean: 204.52031060000309 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.7314033219481595,
            "unit": "iter/sec",
            "range": "stddev: 0.030889934521880272",
            "extra": "mean: 1.3672346980000156 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d1560263be982c04ba923b4fccbce7d61e1df383",
          "message": "MAINT: update pip constraints and pre-commit (#486)\n\n* MAINT: address pyright issues\r\n* MAINT: ignore `numpy` deprecation error\r\n  This is an upstream problem in TensorFlow\r\n* FIX: set initial dynamics on D0, not J/psi\r\n* MAINT: resolve `virtualenv` dependency conflict\r\n* MAINT: use correct inv mass key after QRules v0.9.8\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2023-05-13T12:24:43-04:00",
          "tree_id": "895a62e51c419f10d56f904fd8d90ba6654fed1a",
          "url": "https://github.com/ComPWA/tensorwaves/commit/d1560263be982c04ba923b4fccbce7d61e1df383"
        },
        "date": 1683995358331,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.20874100216221758,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.790625654000053 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2027706479956806,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.931680249999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.1880980889806355,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 5.316375117999996 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.45149505381208066,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.2148636880000367 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 15.559348890241521,
            "unit": "iter/sec",
            "range": "stddev: 0.0009730995457395636",
            "extra": "mean: 64.27004157141677 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 122.40683609579072,
            "unit": "iter/sec",
            "range": "stddev: 0.00018693970853275076",
            "extra": "mean: 8.169478371432131 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.6475530941262484,
            "unit": "iter/sec",
            "range": "stddev: 0.1848692543838172",
            "extra": "mean: 377.7072506000195 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 59.88838179400339,
            "unit": "iter/sec",
            "range": "stddev: 0.0005733964298731248",
            "extra": "mean: 16.69772951020243 msec\nrounds: 49"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.871869657402817,
            "unit": "iter/sec",
            "range": "stddev: 0.0012288807678355273",
            "extra": "mean: 170.3035077999857 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 11.316305809375692,
            "unit": "iter/sec",
            "range": "stddev: 0.0007139067551344572",
            "extra": "mean: 88.36806081817693 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 11.484009766390159,
            "unit": "iter/sec",
            "range": "stddev: 0.001809304309818959",
            "extra": "mean: 87.07759923077255 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8585675657805294,
            "unit": "iter/sec",
            "range": "stddev: 0.018174167038238508",
            "extra": "mean: 1.164730697799996 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.649007079674577,
            "unit": "iter/sec",
            "range": "stddev: 0.001113314426502466",
            "extra": "mean: 177.0222600000011 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 11.732236446397918,
            "unit": "iter/sec",
            "range": "stddev: 0.0018582870029331044",
            "extra": "mean: 85.23524091666464 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 11.569842533613425,
            "unit": "iter/sec",
            "range": "stddev: 0.0015186963286075194",
            "extra": "mean: 86.43159983333722 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.9951773839318847,
            "unit": "iter/sec",
            "range": "stddev: 0.005827561588831199",
            "extra": "mean: 1.0048459863999937 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "62f98babaaa13f253e13fb60123acfe9f544b1d7",
          "message": "MAINT: update pip constraints and pre-commit (#491)\n\n* DOC: update link to Minuit\r\n* MAINT: address mypy issue\r\n* MAINT: run macOS job on Python 3.9\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-06-22T11:22:20+02:00",
          "tree_id": "52a3b9a236b4e687b76eb0ad81420506fdf70454",
          "url": "https://github.com/ComPWA/tensorwaves/commit/62f98babaaa13f253e13fb60123acfe9f544b1d7"
        },
        "date": 1687425964678,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2901922229192896,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.4459917290000135 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2634043549828361,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7964444440000307 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.26553419166286946,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7659933500000307 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4676979156579846,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.138132256999995 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.74270564554082,
            "unit": "iter/sec",
            "range": "stddev: 0.0006585322888863574",
            "extra": "mean: 50.65161877778716 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 143.9141831026176,
            "unit": "iter/sec",
            "range": "stddev: 0.00016229161380205078",
            "extra": "mean: 6.948585458647621 msec\nrounds: 133"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.228721541323787,
            "unit": "iter/sec",
            "range": "stddev: 0.0008387125787069044",
            "extra": "mean: 236.47809159998587 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 81.13828527416798,
            "unit": "iter/sec",
            "range": "stddev: 0.0003016644489536723",
            "extra": "mean: 12.324638074627519 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.738498947293085,
            "unit": "iter/sec",
            "range": "stddev: 0.0006596853057976987",
            "extra": "mean: 129.22402740001644 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.037168994618185,
            "unit": "iter/sec",
            "range": "stddev: 0.00037136490118932396",
            "extra": "mean: 66.50187946666695 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.04144555631897,
            "unit": "iter/sec",
            "range": "stddev: 0.0008291099227213222",
            "extra": "mean: 66.4829717500055 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2189537888236268,
            "unit": "iter/sec",
            "range": "stddev: 0.0021248510486715716",
            "extra": "mean: 820.3756443999964 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.79461225026567,
            "unit": "iter/sec",
            "range": "stddev: 0.0003689615313048316",
            "extra": "mean: 147.1754330000067 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.57812602730316,
            "unit": "iter/sec",
            "range": "stddev: 0.0002016271541983288",
            "extra": "mean: 64.19257350000507 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.281029706913634,
            "unit": "iter/sec",
            "range": "stddev: 0.0009903810614936002",
            "extra": "mean: 65.44061618750519 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.3743877149459258,
            "unit": "iter/sec",
            "range": "stddev: 0.0016611445228898033",
            "extra": "mean: 727.5967247999915 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ac248308b37d74e399c6ea37d950e88a932beb0d",
          "message": "DOC: improve amplitude analysis tutorial (#489)\n\n* DOC: explain rest frame of decaying particle\r\n* DOC: remove `reaction_info` definition\r\n* DOC: show amplitude model expressions\r\n* DOC: remove `max_complexity` argument\r\n* FIX: fix typo \"show(s)\"\r\n* MAINT: avoid using `_` and `__` IPython variables\r\n* MAINT: merge Markdown cells (apart from headings)\r\n* MAINT: merge `src = aslatex` lines\r\n* MAINT: rename `intensity` to `intensity_func`\r\n* MAINT: write MyST cross-references where possible\r\n  MyST references work better with `jupyterlab-myst` than reStructuredText\r\n\r\nCo-authored-by: Lena Poepping <lpoepping@ep1.rub.de>",
          "timestamp": "2023-06-29T07:51:32Z",
          "tree_id": "bc20909af7d3e0152fb08e401ed8e6c3191a0b43",
          "url": "https://github.com/ComPWA/tensorwaves/commit/ac248308b37d74e399c6ea37d950e88a932beb0d"
        },
        "date": 1688025313895,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.28280237014415527,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.536038256999973 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2535105265642394,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.9446093760000167 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2681105718616581,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.7298044350000055 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.516172337452112,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9373374499999727 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.198243058707614,
            "unit": "iter/sec",
            "range": "stddev: 0.00102773336065081",
            "extra": "mean: 52.088099777778204 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 141.52696189151376,
            "unit": "iter/sec",
            "range": "stddev: 0.00022126168756293873",
            "extra": "mean: 7.065791469236379 msec\nrounds: 130"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.4546077530574335,
            "unit": "iter/sec",
            "range": "stddev: 0.11272112594392947",
            "extra": "mean: 289.4684639999923 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 74.62771732916036,
            "unit": "iter/sec",
            "range": "stddev: 0.0003136355160835619",
            "extra": "mean: 13.399847078121141 msec\nrounds: 64"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.032118603993637,
            "unit": "iter/sec",
            "range": "stddev: 0.00042729646424259345",
            "extra": "mean: 142.20465500000046 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 14.869979398297922,
            "unit": "iter/sec",
            "range": "stddev: 0.0016287884243066399",
            "extra": "mean: 67.24958879999956 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 14.951713255145576,
            "unit": "iter/sec",
            "range": "stddev: 0.0011812237821609804",
            "extra": "mean: 66.8819675000023 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.0176766015071355,
            "unit": "iter/sec",
            "range": "stddev: 0.005792336422560557",
            "extra": "mean: 982.6304334000042 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.528231019075733,
            "unit": "iter/sec",
            "range": "stddev: 0.0003708364082594496",
            "extra": "mean: 153.18085359999714 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 14.60869514404637,
            "unit": "iter/sec",
            "range": "stddev: 0.001266803825253382",
            "extra": "mean: 68.45238333332873 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 14.68340738706462,
            "unit": "iter/sec",
            "range": "stddev: 0.0009952509054025735",
            "extra": "mean: 68.10408331249818 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1893432951128937,
            "unit": "iter/sec",
            "range": "stddev: 0.013385626307845153",
            "extra": "mean: 840.8001324000224 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "66853113+pre-commit-ci[bot]@users.noreply.github.com",
            "name": "pre-commit-ci[bot]",
            "username": "pre-commit-ci[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "75ea9a20fede9924e393fdba62997720c9c3137e",
          "message": "DX!: switch to Ruff as linter (#492)\n\n* MAINT: implement updates from pre-commit hooks\r\n* MAINT: update pip constraints and pre-commit\r\n* MAINT: upgrade to Jupyter Lab v4\r\n\r\n---------\r\n\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: Remco de Boer <29308176+redeboer@users.noreply.github.com>",
          "timestamp": "2023-07-07T03:02:45+02:00",
          "tree_id": "f6fba67cb19678af9439bf527ad3e024a7c4ea3d",
          "url": "https://github.com/ComPWA/tensorwaves/commit/75ea9a20fede9924e393fdba62997720c9c3137e"
        },
        "date": 1688691983659,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.271630988823614,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.681465079999981 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.25251712536360127,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.960127450999977 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2524936858411614,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.960495078000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5262701709903311,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.9001646970000365 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.433956058324707,
            "unit": "iter/sec",
            "range": "stddev: 0.0007614092372668653",
            "extra": "mean: 51.45632711110516 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 140.6095892782474,
            "unit": "iter/sec",
            "range": "stddev: 0.0009488940066939008",
            "extra": "mean: 7.111890484376104 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.3128095637504296,
            "unit": "iter/sec",
            "range": "stddev: 0.13154569686511647",
            "extra": "mean: 301.85858280000275 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 74.1804100523143,
            "unit": "iter/sec",
            "range": "stddev: 0.00023966371224991387",
            "extra": "mean: 13.48064804838325 msec\nrounds: 62"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.992591906883337,
            "unit": "iter/sec",
            "range": "stddev: 0.00041924097264745263",
            "extra": "mean: 143.00848860000315 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.095531386363003,
            "unit": "iter/sec",
            "range": "stddev: 0.0008040368992333216",
            "extra": "mean: 66.24476968749704 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.357325874303056,
            "unit": "iter/sec",
            "range": "stddev: 0.0015769746827394666",
            "extra": "mean: 65.11550306250058 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9671459634189454,
            "unit": "iter/sec",
            "range": "stddev: 0.005124445070810917",
            "extra": "mean: 1.0339700912000012 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.677917764978994,
            "unit": "iter/sec",
            "range": "stddev: 0.00022661236444712147",
            "extra": "mean: 149.74727680000797 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 14.988017185173344,
            "unit": "iter/sec",
            "range": "stddev: 0.0002971793292938949",
            "extra": "mean: 66.71996620001437 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 14.980945524716828,
            "unit": "iter/sec",
            "range": "stddev: 0.001010360773538639",
            "extra": "mean: 66.75146093750328 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1212092588803995,
            "unit": "iter/sec",
            "range": "stddev: 0.008266164511257633",
            "extra": "mean: 891.8941688000018 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d4abb887cf81d7116ea4314e8373a97a431e8ab2",
          "message": "MAINT: verify installation on Python 3.11 (#484)\n\n* MAINT: ignore deprecation error in pytest\r\n* MAINT: update pip constraints and pre-commit\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>",
          "timestamp": "2023-07-07T03:24:22+02:00",
          "tree_id": "68a8143443aa8e9126f3e6175577e496a895e646",
          "url": "https://github.com/ComPWA/tensorwaves/commit/d4abb887cf81d7116ea4314e8373a97a431e8ab2"
        },
        "date": 1688693290847,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2702242213607097,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.700630516999979 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2337708174411991,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.277693900999992 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2410983715813829,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.147684588000004 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4694573536205719,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.130118938999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 18.618563299681487,
            "unit": "iter/sec",
            "range": "stddev: 0.0012440659916492002",
            "extra": "mean: 53.709836999995986 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 130.30711187070494,
            "unit": "iter/sec",
            "range": "stddev: 0.00040484522600029983",
            "extra": "mean: 7.674178221310233 msec\nrounds: 122"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.1759976936870036,
            "unit": "iter/sec",
            "range": "stddev: 0.12666237396986485",
            "extra": "mean: 314.86168960000214 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 68.88739604981026,
            "unit": "iter/sec",
            "range": "stddev: 0.0005329888184352661",
            "extra": "mean: 14.516443607143056 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.2675619671691605,
            "unit": "iter/sec",
            "range": "stddev: 0.002033168370120328",
            "extra": "mean: 159.55167339999434 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 13.612485255576171,
            "unit": "iter/sec",
            "range": "stddev: 0.0011934795021854294",
            "extra": "mean: 73.46197121428385 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 13.592947096049144,
            "unit": "iter/sec",
            "range": "stddev: 0.0015623137855224922",
            "extra": "mean: 73.56756359999774 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.8763918479137487,
            "unit": "iter/sec",
            "range": "stddev: 0.015931618271318188",
            "extra": "mean: 1.141042106199984 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.85899559341708,
            "unit": "iter/sec",
            "range": "stddev: 0.0034540819248159134",
            "extra": "mean: 170.67771839998613 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 13.412785308387157,
            "unit": "iter/sec",
            "range": "stddev: 0.00176984432520993",
            "extra": "mean: 74.55572999999406 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 13.585362351190618,
            "unit": "iter/sec",
            "range": "stddev: 0.001231940291754492",
            "extra": "mean: 73.60863657143162 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.013086671075442,
            "unit": "iter/sec",
            "range": "stddev: 0.012237994568464302",
            "extra": "mean: 987.0823775999838 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1994c0cbe9f31ca5f04d66f73e6345897d64a1f1",
          "message": "MAINT: update pip constraints and pre-commit (#493)\n\n* FIX: restrict `phasespace` in Python 3.7\r\n* MAINT: address Ruff nbQA issues\r\n* MAINT: remove deprecated VSCode linting settings\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-08-07T20:29:27Z",
          "tree_id": "fceee99d5ac6d7ac7e7e9fe1bb51be38f94a19c0",
          "url": "https://github.com/ComPWA/tensorwaves/commit/1994c0cbe9f31ca5f04d66f73e6345897d64a1f1"
        },
        "date": 1691440379010,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3017825906823102,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.3136437650000516 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2716616542288815,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.681049513000005 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.27929414388730484,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.580454591999967 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4949460139441907,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.020422372999974 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.608899807029925,
            "unit": "iter/sec",
            "range": "stddev: 0.0002328262260273309",
            "extra": "mean: 50.99725175001879 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.32037294492073,
            "unit": "iter/sec",
            "range": "stddev: 0.00023256459521438846",
            "extra": "mean: 7.229593000000085 msec\nrounds: 132"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.598635815892014,
            "unit": "iter/sec",
            "range": "stddev: 0.10103774076778008",
            "extra": "mean: 277.88307879999365 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 79.72115870825152,
            "unit": "iter/sec",
            "range": "stddev: 0.00013040464408324798",
            "extra": "mean: 12.543721343283677 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.588271129236381,
            "unit": "iter/sec",
            "range": "stddev: 0.0007136554541388255",
            "extra": "mean: 131.7823234000116 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.062003972553944,
            "unit": "iter/sec",
            "range": "stddev: 0.0006513237616318151",
            "extra": "mean: 66.39222787500287 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.135678301450364,
            "unit": "iter/sec",
            "range": "stddev: 0.0007502687765753088",
            "extra": "mean: 66.06905749999825 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.1097341746486786,
            "unit": "iter/sec",
            "range": "stddev: 0.013248170580160368",
            "extra": "mean: 901.1167023999974 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.873385249349473,
            "unit": "iter/sec",
            "range": "stddev: 0.0010738978522284335",
            "extra": "mean: 145.48871680001412 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.36872930546057,
            "unit": "iter/sec",
            "range": "stddev: 0.0004960364637073555",
            "extra": "mean: 65.06718806249623 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.663651007246633,
            "unit": "iter/sec",
            "range": "stddev: 0.000531604980305985",
            "extra": "mean: 63.84207612499537 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2776026104677691,
            "unit": "iter/sec",
            "range": "stddev: 0.005819241991720028",
            "extra": "mean: 782.7159961999996 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "713ee4706d220aae1926358c699960d3f1155393",
          "message": "DOC: add `CITATION.cff` (#494)\n\n* MAINT: update pip constraints and pre-commit\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-08-24T23:34:46+02:00",
          "tree_id": "53af84d009aec9063ecbfcdbcf1e546900769dba",
          "url": "https://github.com/ComPWA/tensorwaves/commit/713ee4706d220aae1926358c699960d3f1155393"
        },
        "date": 1692913144187,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.24959983218904094,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.006412949999998 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.21135751503689257,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.73131982000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.21000245185212618,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 4.761849165000001 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.42378336184033394,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.359696226999972 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 16.953623044462237,
            "unit": "iter/sec",
            "range": "stddev: 0.0008862769761021869",
            "extra": "mean: 58.98444228572381 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 92.81156145094401,
            "unit": "iter/sec",
            "range": "stddev: 0.024084186023176923",
            "extra": "mean: 10.774519729727364 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 2.8767153345800454,
            "unit": "iter/sec",
            "range": "stddev: 0.13627043966099855",
            "extra": "mean: 347.6186843999926 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 61.61703164230807,
            "unit": "iter/sec",
            "range": "stddev: 0.0003248264634993141",
            "extra": "mean: 16.229279037735573 msec\nrounds: 53"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 5.848575798978509,
            "unit": "iter/sec",
            "range": "stddev: 0.0009428402866469838",
            "extra": "mean: 170.98179699999037 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 11.621560726214309,
            "unit": "iter/sec",
            "range": "stddev: 0.0008280923295628769",
            "extra": "mean: 86.04696249999695 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 11.787878162546095,
            "unit": "iter/sec",
            "range": "stddev: 0.0013624288130991498",
            "extra": "mean: 84.83290938460186 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.7812387346595209,
            "unit": "iter/sec",
            "range": "stddev: 0.006792205948986749",
            "extra": "mean: 1.28001845739999 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 5.582618701569379,
            "unit": "iter/sec",
            "range": "stddev: 0.00022273950941768686",
            "extra": "mean: 179.12740479998774 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 11.76694126664417,
            "unit": "iter/sec",
            "range": "stddev: 0.0010683697287373701",
            "extra": "mean: 84.98385241666047 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 11.720960248028737,
            "unit": "iter/sec",
            "range": "stddev: 0.0018686744763574653",
            "extra": "mean: 85.31724183333722 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 0.8986269653580083,
            "unit": "iter/sec",
            "range": "stddev: 0.005795386871764559",
            "extra": "mean: 1.1128088056000025 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f6dc95182db65e1a5b91a557b8e1c989cb5c6fba",
          "message": "DX: enable language navigation on Jupyter Lab (#495)\n\n* MAINT: apply new black formatting\r\n* MAINT: update pip constraints and pre-commit\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-09-09T23:18:00+02:00",
          "tree_id": "a9837b88f80df2a01d0b49a2ee51a5bbe2101863",
          "url": "https://github.com/ComPWA/tensorwaves/commit/f6dc95182db65e1a5b91a557b8e1c989cb5c6fba"
        },
        "date": 1694294494124,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2839258772554152,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.5220459989999995 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.27386321996892504,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.651457834000013 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.28191393921837204,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.54718182000002 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.49830172592861954,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.006816248000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 20.129183108759513,
            "unit": "iter/sec",
            "range": "stddev: 0.0006623803006741189",
            "extra": "mean: 49.67911487500132 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 145.9023236054476,
            "unit": "iter/sec",
            "range": "stddev: 0.00013236565963560534",
            "extra": "mean: 6.8539004402988315 msec\nrounds: 134"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.581510464547337,
            "unit": "iter/sec",
            "range": "stddev: 0.09868401732699648",
            "extra": "mean: 279.2118045999871 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 77.3309807218163,
            "unit": "iter/sec",
            "range": "stddev: 0.000208537356912493",
            "extra": "mean: 12.931427878786542 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 7.336546885643663,
            "unit": "iter/sec",
            "range": "stddev: 0.0015085719965978411",
            "extra": "mean: 136.303906400002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 14.614019543052141,
            "unit": "iter/sec",
            "range": "stddev: 0.0003205215153227539",
            "extra": "mean: 68.427443733331 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.022099973624412,
            "unit": "iter/sec",
            "range": "stddev: 0.00039958722187217484",
            "extra": "mean: 66.5685890625003 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.091083446916763,
            "unit": "iter/sec",
            "range": "stddev: 0.0016911846732024414",
            "extra": "mean: 916.52018260001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.66942013232215,
            "unit": "iter/sec",
            "range": "stddev: 0.0014718089483854325",
            "extra": "mean: 149.93807259999699 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.15198262461542,
            "unit": "iter/sec",
            "range": "stddev: 0.00024751002509153654",
            "extra": "mean: 65.99796375000011 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.532264779981878,
            "unit": "iter/sec",
            "range": "stddev: 0.0008676509297518695",
            "extra": "mean: 64.38211131249894 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2524884239609146,
            "unit": "iter/sec",
            "range": "stddev: 0.0018304750267369248",
            "extra": "mean: 798.4105727999975 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "21a21bad3ad6a8d9b587fc73b0e2fa78daaa1a62",
          "message": "DOC: show kinematic variable expressions (#497)\n\n* MAINT: ignore stackexchange link\r\n  https://github.com/ComPWA/tensorwaves/actions/runs/6273619138/job/17037692006",
          "timestamp": "2023-09-22T13:47:04+02:00",
          "tree_id": "7502aefa1179a253b0ef074e543445ba48f74512",
          "url": "https://github.com/ComPWA/tensorwaves/commit/21a21bad3ad6a8d9b587fc73b0e2fa78daaa1a62"
        },
        "date": 1695383441855,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.2922748837460355,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.4214366530000007 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26080025283311703,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.8343521110000154 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2697882086158346,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.706611215999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5420259242045318,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.8449302059999866 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.60435864402778,
            "unit": "iter/sec",
            "range": "stddev: 0.0007395919943728402",
            "extra": "mean: 51.009064777777745 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 143.1249521180158,
            "unit": "iter/sec",
            "range": "stddev: 0.00013260620385341224",
            "extra": "mean: 6.9869019007631525 msec\nrounds: 131"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.396517443063922,
            "unit": "iter/sec",
            "range": "stddev: 0.11700774740717114",
            "extra": "mean: 294.4192152000028 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 74.82476102348663,
            "unit": "iter/sec",
            "range": "stddev: 0.00013268366087413748",
            "extra": "mean: 13.364559890623795 msec\nrounds: 64"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.862037351543347,
            "unit": "iter/sec",
            "range": "stddev: 0.0006261311964477472",
            "extra": "mean: 145.72931459999836 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 15.091606902802761,
            "unit": "iter/sec",
            "range": "stddev: 0.001597165128720511",
            "extra": "mean: 66.26199625000062 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 15.34321286069892,
            "unit": "iter/sec",
            "range": "stddev: 0.00031312238963395535",
            "extra": "mean: 65.17539768749891 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 0.9781035514325309,
            "unit": "iter/sec",
            "range": "stddev: 0.0028396798333095207",
            "extra": "mean: 1.0223866364000003 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.480866514458539,
            "unit": "iter/sec",
            "range": "stddev: 0.0001713798779105893",
            "extra": "mean: 154.300354399993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 15.246453009664432,
            "unit": "iter/sec",
            "range": "stddev: 0.0002718823501960658",
            "extra": "mean: 65.58902581250337 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 15.543662382731911,
            "unit": "iter/sec",
            "range": "stddev: 0.0009863915622336588",
            "extra": "mean: 64.33490224999616 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.13507372929071,
            "unit": "iter/sec",
            "range": "stddev: 0.001896610800944471",
            "extra": "mean: 881.000039199995 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8ebaccf8f4f958495057e176a3f2833dab971a10",
          "message": "MAINT: update pip constraints and pre-commit (#498)\n\n* DOC: make Colab TOC visible by default\r\n* DX: lint PRs with shared commitlint config\r\n* DX: merge `setup.cfg` into `pyproject.toml`\r\n* DX: switch to `black-jupyter` hook\r\n* DX: remove `.prettierrc`\r\n* DX: remove GitHub Issue templates\r\n* DX: synchronize ComPWA dev environment\r\n\r\n---------\r\n\r\nSigned-off-by: dependabot[bot] <support@github.com>\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2023-10-09T18:09:58+02:00",
          "tree_id": "051dbd5f84601af1ddeccf3fd1df7f3977adce87",
          "url": "https://github.com/ComPWA/tensorwaves/commit/8ebaccf8f4f958495057e176a3f2833dab971a10"
        },
        "date": 1696868021635,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3233772828025382,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.092363172000006 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.2742240341737364,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6466533760000175 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.2724761781264035,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6700456049999843 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.4967049616059236,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.0132675880000193 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.60858266704532,
            "unit": "iter/sec",
            "range": "stddev: 0.0003317511902099308",
            "extra": "mean: 50.9980765555598 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 138.73009729631352,
            "unit": "iter/sec",
            "range": "stddev: 0.00015608434466110298",
            "extra": "mean: 7.2082411782938545 msec\nrounds: 129"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.127918160330613,
            "unit": "iter/sec",
            "range": "stddev: 0.0005373265173938708",
            "extra": "mean: 242.25286479999113 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 76.36168247646248,
            "unit": "iter/sec",
            "range": "stddev: 0.00021250647654599548",
            "extra": "mean: 13.095573166662971 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.763960969320864,
            "unit": "iter/sec",
            "range": "stddev: 0.00047210573234742327",
            "extra": "mean: 147.84236699999838 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.823204928145545,
            "unit": "iter/sec",
            "range": "stddev: 0.000268990257427648",
            "extra": "mean: 92.39407427272474 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.764265950285742,
            "unit": "iter/sec",
            "range": "stddev: 0.000812587174590196",
            "extra": "mean: 92.89997150000318 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.1061397469338166,
            "unit": "iter/sec",
            "range": "stddev: 0.00109517564329215",
            "extra": "mean: 904.0449028000012 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.768257315564038,
            "unit": "iter/sec",
            "range": "stddev: 0.0033235817280395544",
            "extra": "mean: 147.74851979998402 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.64067741461517,
            "unit": "iter/sec",
            "range": "stddev: 0.00042083888932335745",
            "extra": "mean: 103.72715080000603 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.591927510542575,
            "unit": "iter/sec",
            "range": "stddev: 0.0008528343217009689",
            "extra": "mean: 104.25433249999969 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.2520520980510483,
            "unit": "iter/sec",
            "range": "stddev: 0.0069611173895084795",
            "extra": "mean: 798.6888098000122 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "aa36e6636b5179d305a6186a56cc272f3678b52e",
          "message": "ENH: set data keys as first positional arguments (#488)",
          "timestamp": "2023-11-08T14:39:44+01:00",
          "tree_id": "6089e24e11f2b69ed0df9a4b6872c60e8bf1cb39",
          "url": "https://github.com/ComPWA/tensorwaves/commit/aa36e6636b5179d305a6186a56cc272f3678b52e"
        },
        "date": 1699450955473,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.43212293449461875,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.31415627399997 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.35976070162926915,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.7796254439999757 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.3904975984805703,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5608352110000396 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.7198242756916121,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.3892279460000054 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 24.933745065649873,
            "unit": "iter/sec",
            "range": "stddev: 0.0025536172202358055",
            "extra": "mean: 40.10628958333484 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 172.19244861724295,
            "unit": "iter/sec",
            "range": "stddev: 0.00011179184561905289",
            "extra": "mean: 5.807455600000466 msec\nrounds: 160"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 5.384394705612399,
            "unit": "iter/sec",
            "range": "stddev: 0.0015094303684840972",
            "extra": "mean: 185.72189720000551 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 98.72208410799944,
            "unit": "iter/sec",
            "range": "stddev: 0.00042555976493016223",
            "extra": "mean: 10.12944579761936 msec\nrounds: 84"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.352295111672042,
            "unit": "iter/sec",
            "range": "stddev: 0.0005659957504415581",
            "extra": "mean: 106.92562500000236 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.772740356648908,
            "unit": "iter/sec",
            "range": "stddev: 0.0013011033495843871",
            "extra": "mean: 102.32544440000879 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.838227327021672,
            "unit": "iter/sec",
            "range": "stddev: 0.0020212336935661233",
            "extra": "mean: 101.64432745454053 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2942062651987425,
            "unit": "iter/sec",
            "range": "stddev: 0.0019623407254366334",
            "extra": "mean: 772.6743618 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.800143036821254,
            "unit": "iter/sec",
            "range": "stddev: 0.0004603690333550194",
            "extra": "mean: 113.63451659999555 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.531845132517335,
            "unit": "iter/sec",
            "range": "stddev: 0.0023333682864347567",
            "extra": "mean: 104.91148209999324 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.642699573915468,
            "unit": "iter/sec",
            "range": "stddev: 0.0009152341490643514",
            "extra": "mean: 103.70539829998506 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4733997971546289,
            "unit": "iter/sec",
            "range": "stddev: 0.004730094546128211",
            "extra": "mean: 678.7024146000022 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d5d235b8687f60e1e68d7621d807a859f29c440b",
          "message": "BREAK: drop Python 3.7 support (#503)",
          "timestamp": "2023-11-08T15:23:40+01:00",
          "tree_id": "0d9d559c15840b47b08ee1c13c6bc19dfdb95eaf",
          "url": "https://github.com/ComPWA/tensorwaves/commit/d5d235b8687f60e1e68d7621d807a859f29c440b"
        },
        "date": 1699453628049,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.31415973790035123,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.183094073999996 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.26104470971504523,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.83076140899999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.27406293816403154,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 3.6487969030000045 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.5318913710040465,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.8800831420000463 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 19.617707638688213,
            "unit": "iter/sec",
            "range": "stddev: 0.000532246773336842",
            "extra": "mean: 50.974355333336355 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 139.80880106622843,
            "unit": "iter/sec",
            "range": "stddev: 0.00014542859934145723",
            "extra": "mean: 7.152625531251733 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.1803548449568515,
            "unit": "iter/sec",
            "range": "stddev: 0.0010183324966209386",
            "extra": "mean: 239.2141425999739 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 76.54866232668877,
            "unit": "iter/sec",
            "range": "stddev: 0.00019526381491024122",
            "extra": "mean: 13.063585562504976 msec\nrounds: 64"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 6.5763898363734326,
            "unit": "iter/sec",
            "range": "stddev: 0.0008011846807966355",
            "extra": "mean: 152.0591121999928 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.13240216238153,
            "unit": "iter/sec",
            "range": "stddev: 0.0018971397736151576",
            "extra": "mean: 98.69327963636206 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.20940265398729,
            "unit": "iter/sec",
            "range": "stddev: 0.0015073904434401081",
            "extra": "mean: 97.9489235454387 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.000226246375847,
            "unit": "iter/sec",
            "range": "stddev: 0.01792506658036454",
            "extra": "mean: 999.7738047999974 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 6.4209552693502445,
            "unit": "iter/sec",
            "range": "stddev: 0.00011783721696900663",
            "extra": "mean: 155.74006639999425 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.338653464599714,
            "unit": "iter/sec",
            "range": "stddev: 0.001679221392430391",
            "extra": "mean: 107.08181900000113 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.204646641853843,
            "unit": "iter/sec",
            "range": "stddev: 0.0013271112580571693",
            "extra": "mean: 108.64078099999688 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.1537007679364852,
            "unit": "iter/sec",
            "range": "stddev: 0.0022139666123623253",
            "extra": "mean: 866.7758813999967 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8d56f2cbb9c413f64213b4507cdf6163f317d72c",
          "message": "MAINT: update pip constraints and pre-commit (#504)\n\n* DX: switch to faster `black` pre-commit hook\r\n* MAINT: apply new `black` formatting\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-11-08T23:09:58+01:00",
          "tree_id": "c3a47db4ebbb11a2b29fda5bf8f1e2453a636a93",
          "url": "https://github.com/ComPWA/tensorwaves/commit/8d56f2cbb9c413f64213b4507cdf6163f317d72c"
        },
        "date": 1699481572335,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.3816475792785772,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.6202183750000074 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.34770254576274096,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.876021508000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.3929905011661109,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.544590764999981 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.7075756215271952,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.4132765029999916 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 24.073646049847437,
            "unit": "iter/sec",
            "range": "stddev: 0.0006362793036168063",
            "extra": "mean: 41.539200083335004 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 169.20317925788478,
            "unit": "iter/sec",
            "range": "stddev: 0.00009396692809679679",
            "extra": "mean: 5.910054435064053 msec\nrounds: 154"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.277840167985081,
            "unit": "iter/sec",
            "range": "stddev: 0.11189846308961694",
            "extra": "mean: 233.76282440000864 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 98.21588815679524,
            "unit": "iter/sec",
            "range": "stddev: 0.00020892460562434888",
            "extra": "mean: 10.181652060240653 msec\nrounds: 83"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.520722905443499,
            "unit": "iter/sec",
            "range": "stddev: 0.0002764935073545781",
            "extra": "mean: 105.03404100000087 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.986389769084766,
            "unit": "iter/sec",
            "range": "stddev: 0.0003588154127644897",
            "extra": "mean: 100.13628779999522 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.951577226515028,
            "unit": "iter/sec",
            "range": "stddev: 0.002569842273476762",
            "extra": "mean: 100.48658390909085 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2764173050178063,
            "unit": "iter/sec",
            "range": "stddev: 0.004842743131040477",
            "extra": "mean: 783.4428412000022 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.917471072049771,
            "unit": "iter/sec",
            "range": "stddev: 0.00022933801725988137",
            "extra": "mean: 112.13941619999446 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.728340347091923,
            "unit": "iter/sec",
            "range": "stddev: 0.001948000820532059",
            "extra": "mean: 102.79245630000275 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.6778208439193,
            "unit": "iter/sec",
            "range": "stddev: 0.0009689159611375312",
            "extra": "mean: 103.32904649999932 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4837722869675307,
            "unit": "iter/sec",
            "range": "stddev: 0.0015938224536807513",
            "extra": "mean: 673.9578631999905 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "379e96a4eb99b14ab0b542333caa470897fd55e4",
          "message": "MAINT: update pip constraints and pre-commit (#505)\n\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-12-02T17:57:59+01:00",
          "tree_id": "a01dad92e1a0d331fbecc3f5a8d821b9d79b6f7b",
          "url": "https://github.com/ComPWA/tensorwaves/commit/379e96a4eb99b14ab0b542333caa470897fd55e4"
        },
        "date": 1701536454371,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.43351998344026665,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.306698740999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.3620214414084326,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.7622673289999966 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.39674866767978684,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.520487355 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.709063592106619,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.4103107409999893 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 24.64856967996179,
            "unit": "iter/sec",
            "range": "stddev: 0.0008227520521509084",
            "extra": "mean: 40.570305416665065 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 170.35885752628204,
            "unit": "iter/sec",
            "range": "stddev: 0.00021851852682061815",
            "extra": "mean: 5.869961882350176 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 5.3585144875786295,
            "unit": "iter/sec",
            "range": "stddev: 0.003425661398260443",
            "extra": "mean: 186.61888520000502 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 97.12878188278886,
            "unit": "iter/sec",
            "range": "stddev: 0.0005054390072765544",
            "extra": "mean: 10.295609402440155 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.546827029856374,
            "unit": "iter/sec",
            "range": "stddev: 0.0005995583574423657",
            "extra": "mean: 104.74684383331123 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.870364313074425,
            "unit": "iter/sec",
            "range": "stddev: 0.0009057297089250766",
            "extra": "mean: 101.31338299999584 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.951731841525277,
            "unit": "iter/sec",
            "range": "stddev: 0.00040962233190925334",
            "extra": "mean: 100.48502270000199 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2644453044976576,
            "unit": "iter/sec",
            "range": "stddev: 0.001851648246884397",
            "extra": "mean: 790.8606220000024 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.941677324863122,
            "unit": "iter/sec",
            "range": "stddev: 0.000438305504665258",
            "extra": "mean: 111.83584059999703 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.50387553408228,
            "unit": "iter/sec",
            "range": "stddev: 0.002401491955524864",
            "extra": "mean: 105.22023320000926 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.558419029770253,
            "unit": "iter/sec",
            "range": "stddev: 0.0003437729349852069",
            "extra": "mean: 104.6198117999893 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4606351669769317,
            "unit": "iter/sec",
            "range": "stddev: 0.001578339506802254",
            "extra": "mean: 684.6336597999994 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "da38b5e671c1617525935655dde3bc72d5b815ad",
          "message": "DX: remove `figure_formats = [\"svg\"]` statement (#507)\n\n* DOC: plot with SVG where needed",
          "timestamp": "2023-12-02T21:31:13+01:00",
          "tree_id": "8d8aa701263b7fd9495426c1c3e24dbd17582995",
          "url": "https://github.com/ComPWA/tensorwaves/commit/da38b5e671c1617525935655dde3bc72d5b815ad"
        },
        "date": 1701549244589,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.4490813952713488,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.226767820999953 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.3636068625588475,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.750223119999987 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.40763050471493895,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4532020750000356 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.714564387761717,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.3994540129999677 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 25.939568457661586,
            "unit": "iter/sec",
            "range": "stddev: 0.00027541674177900184",
            "extra": "mean: 38.55114250000705 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 170.17494080236298,
            "unit": "iter/sec",
            "range": "stddev: 0.00029720799702435827",
            "extra": "mean: 5.8763058490578555 msec\nrounds: 159"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.362210925082705,
            "unit": "iter/sec",
            "range": "stddev: 0.10470255082422625",
            "extra": "mean: 229.24155140000266 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 95.75564450854823,
            "unit": "iter/sec",
            "range": "stddev: 0.0006786672698816303",
            "extra": "mean: 10.443248595237941 msec\nrounds: 84"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.249125472064113,
            "unit": "iter/sec",
            "range": "stddev: 0.0017857324212486574",
            "extra": "mean: 108.11832999999638 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.705133349284818,
            "unit": "iter/sec",
            "range": "stddev: 0.0021907046726763724",
            "extra": "mean: 103.03825449999522 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.77412408856039,
            "unit": "iter/sec",
            "range": "stddev: 0.0022319249689635797",
            "extra": "mean: 102.31095809090428 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2680259407178114,
            "unit": "iter/sec",
            "range": "stddev: 0.002676557762515703",
            "extra": "mean: 788.6273993999794 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.901342888277512,
            "unit": "iter/sec",
            "range": "stddev: 0.0005747430298513004",
            "extra": "mean: 112.34259959999235 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.72572622442682,
            "unit": "iter/sec",
            "range": "stddev: 0.0002499401362877193",
            "extra": "mean: 102.82008529999871 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.65750430679057,
            "unit": "iter/sec",
            "range": "stddev: 0.0003970488454905921",
            "extra": "mean: 103.54642030000036 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4699796598238082,
            "unit": "iter/sec",
            "range": "stddev: 0.0008940639468483801",
            "extra": "mean: 680.2815218000092 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "62d29383ed9eb18088e6ea5241c8a00a9c46d28a",
          "message": "DX: install Ruff and Git in Jupyter Lab (#508)\n\n* MAINT: update pip constraints and pre-commit\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-12-02T22:19:33+01:00",
          "tree_id": "d5fa8328cb0295aa075e0862b0f93b8a8bffa924",
          "url": "https://github.com/ComPWA/tensorwaves/commit/62d29383ed9eb18088e6ea5241c8a00a9c46d28a"
        },
        "date": 1701552145986,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.437084524797518,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.287886995000008 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.3587680066375579,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.787316543000003 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.4017203539178502,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4892938339999944 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.7036156305195318,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.4212305079999794 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 24.188412357226674,
            "unit": "iter/sec",
            "range": "stddev: 0.0007534289260663176",
            "extra": "mean: 41.34210981818465 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 170.81635433677582,
            "unit": "iter/sec",
            "range": "stddev: 0.00010511149391413387",
            "extra": "mean: 5.854240385135685 msec\nrounds: 148"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 5.381300112980642,
            "unit": "iter/sec",
            "range": "stddev: 0.002969356652658538",
            "extra": "mean: 185.82869919999894 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 96.9177858265331,
            "unit": "iter/sec",
            "range": "stddev: 0.00015276422257347528",
            "extra": "mean: 10.318023585369929 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.442822957860178,
            "unit": "iter/sec",
            "range": "stddev: 0.00047252333395866875",
            "extra": "mean: 105.9005346666595 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.91638664418846,
            "unit": "iter/sec",
            "range": "stddev: 0.0004688128888358723",
            "extra": "mean: 100.84318369998755 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.961051771410316,
            "unit": "iter/sec",
            "range": "stddev: 0.0010604919975808595",
            "extra": "mean: 100.39100518181695 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2667642385403302,
            "unit": "iter/sec",
            "range": "stddev: 0.0018280588552106489",
            "extra": "mean: 789.4128753999894 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.775124909119128,
            "unit": "iter/sec",
            "range": "stddev: 0.0028395968543339254",
            "extra": "mean: 113.9584918000196 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.703873736129449,
            "unit": "iter/sec",
            "range": "stddev: 0.00031008631549697577",
            "extra": "mean: 103.05162939999946 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.577039768880669,
            "unit": "iter/sec",
            "range": "stddev: 0.0021831815113605704",
            "extra": "mean: 104.41639839999084 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.457207519818939,
            "unit": "iter/sec",
            "range": "stddev: 0.0038842093149245743",
            "extra": "mean: 686.2440567999897 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5a9c6869a6f2c2ec5fd1e65c387eae2181628900",
          "message": "MAINT: update pip constraints and pre-commit (#509)\n\n* DX: gitignore `.jupyter_ystore.db`\r\n  This is a file produced by `jupyter-collaboration`\r\n* DX: gitignore `oryx-build-commands.txt`\r\n  Produced when running on GitHub Codespaces\r\n* MAINT: clean up `conf.py` with new Sphinx extensions\r\n* MAINT: remove `ypy-websocket` version constraints\r\n\r\n---------\r\n\r\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-12-09T12:15:16Z",
          "tree_id": "5ff389e1af785eecf390b8a9ee10e5745a3324b9",
          "url": "https://github.com/ComPWA/tensorwaves/commit/5a9c6869a6f2c2ec5fd1e65c387eae2181628900"
        },
        "date": 1702124284430,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.4490476097380106,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.226935358999981 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.36424331565089807,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.745417573999987 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.3597566883779802,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.7796564519999833 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.7186175206817564,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.3915608389999932 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 25.308863944722617,
            "unit": "iter/sec",
            "range": "stddev: 0.0008041049956965465",
            "extra": "mean: 39.51184858333079 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 170.37527896102978,
            "unit": "iter/sec",
            "range": "stddev: 0.00031463388509224183",
            "extra": "mean: 5.869396112500169 msec\nrounds: 160"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 4.343106697271616,
            "unit": "iter/sec",
            "range": "stddev: 0.10712633480185386",
            "extra": "mean: 230.24992699999984 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 99.53932253536219,
            "unit": "iter/sec",
            "range": "stddev: 0.00010964246578115501",
            "extra": "mean: 10.046280952381824 msec\nrounds: 84"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.614157291609947,
            "unit": "iter/sec",
            "range": "stddev: 0.00041062018716102447",
            "extra": "mean: 104.01327642857235 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.880438678949124,
            "unit": "iter/sec",
            "range": "stddev: 0.002308488345340919",
            "extra": "mean: 101.2100810999982 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 10.079784444506068,
            "unit": "iter/sec",
            "range": "stddev: 0.0011353829737142304",
            "extra": "mean: 99.20847072727281 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2794051782076015,
            "unit": "iter/sec",
            "range": "stddev: 0.0021737250572103936",
            "extra": "mean: 781.6132192000055 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.954010491898853,
            "unit": "iter/sec",
            "range": "stddev: 0.00023688871095418616",
            "extra": "mean: 111.6817989999845 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.718000061630772,
            "unit": "iter/sec",
            "range": "stddev: 0.0003629529757813119",
            "extra": "mean: 102.90183100000831 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.647362039890124,
            "unit": "iter/sec",
            "range": "stddev: 0.0018035583797579957",
            "extra": "mean: 103.65527860001293 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4925474845870326,
            "unit": "iter/sec",
            "range": "stddev: 0.0014278888515757765",
            "extra": "mean: 669.9954341999955 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f99a8aed1332309c045d8b8fadd795e71851de23",
          "message": "DX: define `docnblive` job in `tox.ini` (#510)\n\n* DX: activate VSCode multi-file diff editor\r\n  https://code.visualstudio.com/updates/v1_85\\#_multifile-diff-editor\r\n* DX: run mypy over more libraries",
          "timestamp": "2023-12-09T20:53:18+01:00",
          "tree_id": "d92b54c62d0aafdeb796fbead6e110a990207b8f",
          "url": "https://github.com/ComPWA/tensorwaves/commit/f99a8aed1332309c045d8b8fadd795e71851de23"
        },
        "date": 1702151770010,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.4465261716839184,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.239510387999985 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.36386370236975346,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.7482818249999923 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.4013813038252825,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4913965609999877 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.7123784339915704,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.4037482779999948 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 24.357444362289208,
            "unit": "iter/sec",
            "range": "stddev: 0.0015561613728132227",
            "extra": "mean: 41.055210272725674 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 168.92628350306603,
            "unit": "iter/sec",
            "range": "stddev: 0.0003475551488563301",
            "extra": "mean: 5.919741909090482 msec\nrounds: 154"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 5.454426304090271,
            "unit": "iter/sec",
            "range": "stddev: 0.0011835610331016402",
            "extra": "mean: 183.33733820000475 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 96.28515062471953,
            "unit": "iter/sec",
            "range": "stddev: 0.00027224468343757405",
            "extra": "mean: 10.385817475610486 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.494113443959419,
            "unit": "iter/sec",
            "range": "stddev: 0.0005838068514484368",
            "extra": "mean: 105.32842333332819 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 9.947875302053593,
            "unit": "iter/sec",
            "range": "stddev: 0.0010936996841148345",
            "extra": "mean: 100.52397820000465 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.757229131284907,
            "unit": "iter/sec",
            "range": "stddev: 0.001281990531091142",
            "extra": "mean: 102.48811281818409 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2466274976955207,
            "unit": "iter/sec",
            "range": "stddev: 0.0064074819794267615",
            "extra": "mean: 802.1642405999955 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.903490296406224,
            "unit": "iter/sec",
            "range": "stddev: 0.00025388532167912797",
            "extra": "mean: 112.31550400000287 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.710119425525837,
            "unit": "iter/sec",
            "range": "stddev: 0.0003645638402335564",
            "extra": "mean: 102.98534510000081 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.660076466866018,
            "unit": "iter/sec",
            "range": "stddev: 0.0010043942136489395",
            "extra": "mean: 103.5188493000021 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4318513057195952,
            "unit": "iter/sec",
            "range": "stddev: 0.0058390169213783545",
            "extra": "mean: 698.3965415999933 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29308176+redeboer@users.noreply.github.com",
            "name": "Remco de Boer",
            "username": "redeboer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fb4dcadeadd06505c5d4bc4635a4675e8a17b6d8",
          "message": "MAINT: update pip constraints and pre-commit (#512)\n\nCo-authored-by: GitHub <noreply@github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2024-01-12T18:58:47+01:00",
          "tree_id": "8bdfe53d3c289d574a6b99f4692fc52878e131e2",
          "url": "https://github.com/ComPWA/tensorwaves/commit/fb4dcadeadd06505c5d4bc4635a4675e8a17b6d8"
        },
        "date": 1705082564438,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-jax]",
            "value": 0.39675265301367196,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.5204620370000157 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-numpy]",
            "value": 0.3585889637018703,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.788708245999999 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_data[10000-tf]",
            "value": 0.40080764118477247,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.4949624139999855 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/ampform.py::TestJPsiToGammaPiPi::test_fit[10000-jax]",
            "value": 0.7035757906282603,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 1.421310984999991 sec\nrounds: 1"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-jax]",
            "value": 22.998627209851783,
            "unit": "iter/sec",
            "range": "stddev: 0.0003338202511010273",
            "extra": "mean: 43.48085609090772 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numpy]",
            "value": 165.64147369408923,
            "unit": "iter/sec",
            "range": "stddev: 0.0001394364806721279",
            "extra": "mean: 6.037135372550625 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-numba]",
            "value": 3.979907310585112,
            "unit": "iter/sec",
            "range": "stddev: 0.14261125979147188",
            "extra": "mean: 251.2621330000229 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_data[3000-tf]",
            "value": 90.4048028509653,
            "unit": "iter/sec",
            "range": "stddev: 0.00040324760504923654",
            "extra": "mean: 11.061359225001866 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-jax]",
            "value": 9.223036353304794,
            "unit": "iter/sec",
            "range": "stddev: 0.005165543057354368",
            "extra": "mean: 108.42416333333442 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numpy]",
            "value": 10.000612097464018,
            "unit": "iter/sec",
            "range": "stddev: 0.00044786861452491807",
            "extra": "mean: 99.9938794000002 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-numba]",
            "value": 9.804248840046089,
            "unit": "iter/sec",
            "range": "stddev: 0.001175691257319952",
            "extra": "mean: 101.99659518181906 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-Minuit2-tf]",
            "value": 1.2547548647004356,
            "unit": "iter/sec",
            "range": "stddev: 0.0044715362398509385",
            "extra": "mean: 796.9684183999902 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-jax]",
            "value": 8.803033459130932,
            "unit": "iter/sec",
            "range": "stddev: 0.0008016139245626364",
            "extra": "mean: 113.59720540000353 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numpy]",
            "value": 9.516494683765178,
            "unit": "iter/sec",
            "range": "stddev: 0.0031013610523642344",
            "extra": "mean: 105.08070810000731 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-numba]",
            "value": 9.43211450402038,
            "unit": "iter/sec",
            "range": "stddev: 0.0017433372251874184",
            "extra": "mean: 106.0207655000113 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/expression.py::test_fit[1000-ScipyMinimizer-tf]",
            "value": 1.4213881798386767,
            "unit": "iter/sec",
            "range": "stddev: 0.0032765754891254137",
            "extra": "mean: 703.5375797999791 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}