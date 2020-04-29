import tensorflow as tf
import logging

# copied from TensorFlowAnalysis


def _relativistic_breit_wigner(s, mass, width):
    """
    Relativistic Breit-Wigner
    """
    if width.dtype is tf.complex128:
        return tf.math.reciprocal(
            tf.cast(mass * mass - s)
            - tf.complex(tf.constant(0.0), mass) * width
        )
    if width.dtype is tf.float64:
        return tf.math.reciprocal(tf.complex(mass * mass - s, -mass * width))
    return None


# copied from TensorFlowAnalysis


def _wigner_d(phi, theta, psi, j, m1, m2):
    """Calculate Wigner capital-D function.
      phi,
      theta,
      psi  : Rotation angles
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2, e.g. 1 for projection 1/2)

    :param phi:
    :param theta:
    :param psi:
    :param j2:
    :param m2_1:
    :param m2_2:

    """
    i = tf.complex(Const(0), Const(1))
    return (
        tf.exp(-i * tf.cast(m1 / 2.0 * phi, tf.complex128))
        * tf.cast(Wignerd(theta, j, m1, m2), tf.complex128)
        * tf.exp(-i * tf.cast(m2 / 2.0 * psi, tf.complex128))
    )


# copied from TensorFlowAnalysis


def _wigner_d_small(theta, j, m1, m2):
    """Calculate Wigner small-d function. Needs sympy.
      theta : angle
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2)

    :param theta:
    :param j:
    :param m1:
    :param m2:

    """
    from sympy import Rational
    from sympy.abc import x
    from sympy.utilities.lambdify import lambdify
    from sympy.physics.quantum.spin import Rotation as Wigner

    d = (
        Wigner.d(Rational(j, 2), Rational(m1, 2), Rational(m2, 2), x)
        .doit()
        .evalf()
    )
    return lambdify(x, d, "tensorflow")(theta)


def create_intensity(recipe: dict):
    if "Intensity" in recipe:
        recipe = recipe["Intensity"]

    # return _TFNormalizedIntensity(_create_intensity(recipe), None)
    return _create_intensity(recipe)


def _create_intensity(recipe: dict):
    intensity_class = recipe["Class"]
    logging.info("creating", intensity_class)

    # this dict could be used for builder injections later on
    known_intensity_builders = {
        "IncoherentIntensity": _IncoherentIntensity,
        "CoherentIntensity": _CoherentIntensity,
        "StrengthIntensity": _StrengthIntensity,
        "NormalizedIntensity": _NormalizedIntensity,
    }
    if intensity_class in known_intensity_builders:
        return known_intensity_builders[intensity_class](recipe)
    else:
        logging.error(
            "unknown intensity " + str(intensity_class) + "! Skipping"
        )


def _create_amplitude(recipe: dict):
    amplitude_class = recipe["Class"]
    logging.info("creating", amplitude_class)

    known_amplitude_builders = {
        "CoefficientAmplitude": _CoefficientAmplitude,
        "SequentialAmplitude": _SequentialAmplitude,
        "HelicityDecay": _HelicityDecay,
    }
    if amplitude_class in known_amplitude_builders:
        return known_amplitude_builders[amplitude_class](recipe)
    else:
        logging.error(
            "unknown intensity " + str(amplitude_class) + "! Skipping"
        )


class _IncoherentIntensity(tf.keras.Model):
    def __init__(self, recipe: dict):
        super(_IncoherentIntensity, self).__init__(name="IncoherentIntensity")
        if isinstance(recipe["Intensities"], list):
            intensity_recipes = [v for v in recipe["Intensities"]]
            self.intensities = [
                _create_intensity(x) for x in intensity_recipes
            ]
        else:
            raise Exception(
                "Incoherent Intensity requires a list of intensities!"
            )

    def call(self, x):
        return tf.add_n([y(x) for y in self.intensities])


class _CoherentIntensity(tf.keras.Model):
    def __init__(self, recipe: dict):
        super(_CoherentIntensity, self).__init__(name="CoherentIntensity")
        if isinstance(recipe["Amplitudes"], list):
            amp_recipes = [v for v in recipe["Amplitudes"]]
            self.amps = [_create_amplitude(x) for x in amp_recipes]
        else:
            raise Exception(
                "Coherent Intensity requires a list of intensities!"
            )

    def call(self, x):
        return tf.cast(
            tf.pow(
                tf.add_n([amp(x) for amp in self.amps]),
                tf.complex(
                    tf.constant(2.0, dtype=tf.float64),
                    tf.constant(0.0, dtype=tf.float64),
                ),
            ),
            dtype=tf.float64,
        )


class _StrengthIntensity(tf.keras.Model):
    def __init__(self, recipe: dict):
        super(_StrengthIntensity, self).__init__(name="StrengthIntensity")
        if isinstance(recipe, dict):
            self.strength = _register_parameters(recipe["Strength"])
            self.intensity = _create_intensity(recipe["Intensity"])

    def call(self, x):
        return tf.multiply(self.strength, self.intensity(x))


class _NormalizedIntensity(tf.keras.Model):
    def __init__(self, recipe: dict, norm_data, phsp_volume=1.0):
        super(_NormalizedIntensity, self).__init__(name="NormalizedIntensity")
        self.norm_data = norm_data
        self.model = _create_intensity(recipe["Intensity"])
        self.phsp_volume = phsp_volume

    def call(self, x):
        normalization = tf.multiply(
            tf.constant(self.phsp_volume, dtype=tf.float64),
            tf.reduce_sum(self.model(self.norm_data)),
        )
        normalization = tf.divide(normalization, len(self.norm_data))
        return tf.divide(self.model(x), normalization)


class _CoefficientAmplitude(tf.keras.Model):
    def __init__(self, recipe: dict):
        super(_CoefficientAmplitude, self).__init__(
            name="CoefficientAmplitude"
        )
        if isinstance(recipe, dict):
            _register_parameters(recipe["Parameters"])
            self.coefficient = tf.complex(
                tf.Variable(1.0, name="magnitude", dtype=tf.float64),
                tf.Variable(0.0, name="phase", dtype=tf.float64),
            )
            self.amp = _create_amplitude(recipe["Amplitude"])

    def call(self, x):
        return tf.multiply(self.coefficient, self.amp(x))


class _SequentialAmplitude(tf.keras.Model):
    def __init__(self, recipe: list):
        super(_SequentialAmplitude, self).__init__(name="SequentialAmplitude")
        if isinstance(recipe["Amplitudes"], list):
            amp_recipes = [v for v in recipe["Amplitudes"]]
            if len(amp_recipes) == 0:
                raise Exception(
                    "Sequential Amplitude requires a non-empty list of amplitudes!"
                )
            self.seq_amp = [_create_amplitude(x) for x in amp_recipes]
        else:
            raise Exception(
                "Sequential Amplitude requires a list of amplitudes!"
            )

    def call(self, x):
        seq_amp = self.seq_amp[0](x)
        for amp in self.seq_amp[1:]:
            seq_amp = tf.multiply(seq_amp, amp(x))
        return seq_amp


class _HelicityDecay(tf.keras.Model):
    def __init__(self, recipe: dict):
        super(_HelicityDecay, self).__init__(name="HelicityDecay")
        if isinstance(recipe, dict):
            # call a register variable hook here that creates a new variable if it does not exist
            # or if it exists simply returns that (thats how we can couple the parameters)
            self.mass = tf.Variable(1.0, name="mass", dtype=tf.float64)
            self.width = tf.Variable(0.1, name="width", dtype=tf.float64)
            # also we should register kinematic variables here (with a string name)
            # that will be handed down in the call method variable x.
            # then we can do x[kin_var_name] to retrieve it
        else:
            raise Exception("Helicity Decay expects a dictionary recipe!")

    def call(self, x):
        return _relativistic_breit_wigner(x["test"], self.mass, self.width)


def _register_parameters(recipe: dict):
    return tf.Variable(1.0, name="parameter", dtype=tf.float64)
