def get_requirement_groups(requirement):  # pylint: disable=too-many-return-statements
    requirement_lower = requirement.lower()
    if 'bodypix' in requirement_lower:
        return ['bodypix']
    if 'pyfakewebcam' in requirement_lower:
        return ['webcam']
    if 'pafy' in requirement_lower:
        return ['youtube']
    if 'youtube_dl' in requirement_lower:
        return ['youtube']
    if 'mss' in requirement_lower:
        return ['mss']
    if 'mediapipe' in requirement_lower:
        return ['mediapipe']
    return [None]


def get_requirements_with_groups(all_required_packages):
    return [
        (requirement, get_requirement_groups(requirement))
        for requirement in all_required_packages
    ]


def get_required_and_extras(required_packages_with_groups, include_all=True):
    grouped_extras = {}
    all_groups = ['all'] if include_all else []
    for requirement, groups in required_packages_with_groups:
        for group in groups + all_groups:
            grouped_extras.setdefault(group, []).append(requirement)
    return (
        grouped_extras.get(None, []),
        {key: value for key, value in grouped_extras.items() if key}
    )
