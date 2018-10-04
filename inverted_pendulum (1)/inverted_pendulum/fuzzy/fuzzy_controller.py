from .fuzzyinference import FuzzyControl
from math import pi


def get_controller():
    ctrl = FuzzyControl()
    theta_division = 12
    theta_memberships = [
        ['trapezoid_left', -1.57 / theta_division, -0.58 / theta_division, 'vn'],
        ['triangle', -0.3927 / theta_division, 0.327 * 2.0 / theta_division, 'mn'],
        ['triangle', 0.0, 0.1745 * 2.0 / theta_division, 'z'],
        ['triangle', 0.3927 / theta_division, 0.327 * 2.0 / theta_division, 'mp'],
        ['trapezoid_right', 0.58 / theta_division, 1.57 / theta_division, 'vp']
    ]
    ctrl.add_input('theta', (-4*pi, 4 * pi), theta_memberships)

    dtheta_division = 12
    dtheta_memberships = [
        ['trapezoid_left', -1.57 / dtheta_division, -0.58 / dtheta_division, 'vn'],
        ['triangle', -0.3927 / dtheta_division, 2 * 0.327 / dtheta_division, 'mn'],
        ['triangle', 0.0, 2 * 0.1745 / dtheta_division, 'z'],
        ['triangle', 0.3927 / dtheta_division, 2 * 0.327 / dtheta_division, 'mp'],
        ['trapezoid_right', 0.58 / dtheta_division, 1.57 / dtheta_division, 'vp']
    ]
    ctrl.add_input('dtheta', (-1 * 200, 200), dtheta_memberships)

    force_memberships = [
            ['trapezoid_left', -0.99, -0.75, 'vn'],
            ['triangle', -0.6, 0.4, 'sn'],
            ['triangle', -0.3, 0.4, 'mn'],
            ['triangle', 0.0, 0.4, 'z'],
            ['triangle', 0.3, 0.4, 'mp'],
            ['triangle', 0.6, 0.4, 'sp'],
            ['trapezoid_right', 0.75, 0.99, 'vp']
    ]
    ctrl.add_output('force', (-1, 1), force_memberships)

    ctrl.add_rule({'theta': 'vn', 'dtheta': 'vn'}, {'force': 'vp'})
    ctrl.add_rule({'theta': 'vn', 'dtheta': 'mn'}, {'force': 'vp'})
    ctrl.add_rule({'theta': 'vn', 'dtheta': 'z'}, {'force': 'vp'})
    ctrl.add_rule({'theta': 'vn', 'dtheta': 'mp'}, {'force': 'vp'})
    ctrl.add_rule({'theta': 'vn', 'dtheta': 'vp'}, {'force': 'vp'})

    ctrl.add_rule({'theta': 'vp', 'dtheta': 'vn'}, {'force': 'vn'})
    ctrl.add_rule({'theta': 'vp', 'dtheta': 'mn'}, {'force': 'vn'})
    ctrl.add_rule({'theta': 'vp', 'dtheta': 'z'}, {'force': 'vn'})
    ctrl.add_rule({'theta': 'vp', 'dtheta': 'mp'}, {'force': 'vn'})
    ctrl.add_rule({'theta': 'vp', 'dtheta': 'vp'}, {'force': 'vn'})

    ctrl.add_rule({'theta': 'mn', 'dtheta': 'vn'}, {'force': 'vp'})
    ctrl.add_rule({'theta': 'mn', 'dtheta': 'mn'}, {'force': 'sp'})
    ctrl.add_rule({'theta': 'mn', 'dtheta': 'z'}, {'force': 'mp'})
    ctrl.add_rule({'theta': 'mn', 'dtheta': 'mp'}, {'force': 'mp'})
    ctrl.add_rule({'theta': 'mn', 'dtheta': 'vp'}, {'force': 'z'})

    ctrl.add_rule({'theta': 'z', 'dtheta': 'vn'}, {'force': 'sp'})
    ctrl.add_rule({'theta': 'z', 'dtheta': 'mn'}, {'force': 'mp'})
    ctrl.add_rule({'theta': 'z', 'dtheta': 'z'}, {'force': 'z'})
    ctrl.add_rule({'theta': 'z', 'dtheta': 'mp'}, {'force': 'mn'})
    ctrl.add_rule({'theta': 'z', 'dtheta': 'vp'}, {'force': 'sn'})

    ctrl.add_rule({'theta': 'mp', 'dtheta': 'vn'}, {'force': 'z'})
    ctrl.add_rule({'theta': 'mp', 'dtheta': 'mn'}, {'force': 'mn'})
    ctrl.add_rule({'theta': 'mp', 'dtheta': 'z'}, {'force': 'mn'})
    ctrl.add_rule({'theta': 'mp', 'dtheta': 'mp'}, {'force': 'sn'})
    ctrl.add_rule({'theta': 'mp', 'dtheta': 'vp'}, {'force': 'vn'})

    return ctrl
