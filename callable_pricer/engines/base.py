import abc


class PricingEngine(abc.ABC):
    """Abstract interface for manual engines.

    Each engine returns:
    - dirty price (float)
    - state_cache (dict or None): optional state reused across bumps (CRN / geometry)

    The `bond_data` is a plain dict (see CallableBondSpec.to_engine_bond_data).
    """

    @abc.abstractmethod
    def price(self, ts_handle, bond_data, params, state_cache=None):
        raise NotImplementedError
