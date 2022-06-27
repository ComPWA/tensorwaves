window.BENCHMARK_DATA = {
  "lastUpdate": 1656340452609,
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
      }
    ]
  }
}