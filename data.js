window.BENCHMARK_DATA = {
  "lastUpdate": 1645271410663,
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
      }
    ]
  }
}