"""Safety predicate tests use fabricated interface states, not offline proof."""

import unittest

from scripts.jetson_offline_replay_pilot import disconnected


class OfflinePredicateTests(unittest.TestCase):
    def test_only_no_upstream_interfaces_counts_as_disconnected(self):
        state = {name: {'returncode': 0, 'stdout': ''} for name in
                 ('wifi', 'default_route', 'wifi_ipv4', 'ethernet_ipv4')}
        state['wifi']['stdout'] = '30 (disconnected)'
        self.assertTrue(disconnected(state))
        for name, value in (('wifi', '100 (connected)'),
                            ('default_route', 'default via 172.16.88.1 dev wlan0'),
                            ('wifi_ipv4', 'wlan0 inet 172.16.89.139/23'),
                            ('ethernet_ipv4', 'eth0 inet 192.168.1.2/24')):
            changed = {key: row.copy() for key, row in state.items()}
            changed[name]['stdout'] = value
            with self.subTest(name=name):
                self.assertFalse(disconnected(changed))


if __name__ == '__main__':
    unittest.main()
