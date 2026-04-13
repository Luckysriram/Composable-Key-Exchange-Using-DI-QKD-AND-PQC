"""
Layer 8: Network Simulator

Provides Mininet-based network topology simulation for testing
CuKEM in realistic network conditions.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import logging

# Note: Mininet is Linux-only. This module provides the interface
# and can be conditionally imported on Linux systems.

logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Network node configuration"""
    name: str
    ip: str
    role: str  # "client", "server", "router"


@dataclass
class LinkConfig:
    """Network link configuration"""
    node1: str
    node2: str
    bandwidth: str = "100Mbps"  # Link bandwidth
    delay: str = "10ms"         # Propagation delay
    loss: float = 0.0           # Packet loss percentage (0-100)
    jitter: str = "0ms"         # Jitter


@dataclass
class TopologyConfig:
    """Network topology configuration"""
    name: str
    nodes: List[NodeConfig]
    links: List[LinkConfig]
    controller_ip: str = "127.0.0.1"


class NetworkSimulator:
    """
    Mininet-based network simulator for CuKEM testing.

    Provides:
    - Custom network topologies
    - Link characteristics (bandwidth, delay, loss)
    - Failure injection
    - Performance measurement
    """

    def __init__(self, topology_config: Optional[TopologyConfig] = None):
        """
        Initialize network simulator.

        Args:
            topology_config: Network topology configuration
        """
        self.config = topology_config
        self.net = None
        self.nodes: Dict[str, any] = {}
        self.mininet_available = self._check_mininet()

        logger.info("Initialized network simulator")

    def _check_mininet(self) -> bool:
        """Check if Mininet is available"""
        try:
            from mininet.net import Mininet
            return True
        except ImportError:
            logger.warning("Mininet not available (Linux-only)")
            return False

    def create_topology(self, config: Optional[TopologyConfig] = None):
        """
        Create network topology.

        Args:
            config: Topology configuration (uses stored config if None)
        """
        if not self.mininet_available:
            logger.error("Cannot create topology: Mininet not available")
            return

        from mininet.net import Mininet
        from mininet.node import OVSController
        from mininet.link import TCLink
        from mininet.cli import CLI

        config = config or self.config
        if not config:
            raise ValueError("No topology configuration provided")

        logger.info(f"Creating topology: {config.name}")

        # Create Mininet network
        self.net = Mininet(controller=OVSController, link=TCLink)

        # Add controller
        self.net.addController('c0')

        # Add nodes
        for node_config in config.nodes:
            if node_config.role == "client" or node_config.role == "server":
                node = self.net.addHost(
                    node_config.name,
                    ip=node_config.ip
                )
            elif node_config.role == "router":
                node = self.net.addSwitch(node_config.name)
            else:
                raise ValueError(f"Unknown node role: {node_config.role}")

            self.nodes[node_config.name] = node
            logger.debug(f"Added node: {node_config.name} ({node_config.role})")

        # Add links
        for link_config in config.links:
            node1 = self.nodes[link_config.node1]
            node2 = self.nodes[link_config.node2]

            self.net.addLink(
                node1, node2,
                bw=self._parse_bandwidth(link_config.bandwidth),
                delay=link_config.delay,
                loss=link_config.loss,
                jitter=link_config.jitter
            )

            logger.debug(f"Added link: {link_config.node1} <-> {link_config.node2} "
                        f"(bw={link_config.bandwidth}, delay={link_config.delay})")

        logger.info(f"Topology created: {len(self.nodes)} nodes, {len(config.links)} links")

    def start(self):
        """Start network simulation"""
        if not self.net:
            raise RuntimeError("Topology not created")

        logger.info("Starting network")
        self.net.start()
        logger.info("Network started")

    def stop(self):
        """Stop network simulation"""
        if self.net:
            logger.info("Stopping network")
            self.net.stop()
            logger.info("Network stopped")

    def get_node(self, name: str):
        """Get node by name"""
        return self.nodes.get(name)

    def run_command(self, node_name: str, command: str) -> str:
        """
        Run command on a node.

        Args:
            node_name: Node name
            command: Command to execute

        Returns:
            Command output
        """
        node = self.get_node(node_name)
        if not node:
            raise ValueError(f"Node not found: {node_name}")

        logger.debug(f"Running on {node_name}: {command}")
        output = node.cmd(command)
        return output

    def measure_latency(self, src: str, dst: str, count: int = 10) -> Dict:
        """
        Measure latency between nodes using ping.

        Args:
            src: Source node name
            dst: Destination node name
            count: Number of ping packets

        Returns:
            Dictionary with latency statistics
        """
        src_node = self.get_node(src)
        dst_node = self.get_node(dst)

        if not src_node or not dst_node:
            raise ValueError("Node not found")

        dst_ip = dst_node.IP()
        output = src_node.cmd(f"ping -c {count} {dst_ip}")

        # Parse ping output (simplified)
        logger.info(f"Latency {src} -> {dst}:\n{output}")

        return {
            "src": src,
            "dst": dst,
            "count": count,
            "output": output
        }

    def measure_bandwidth(self, src: str, dst: str) -> Dict:
        """
        Measure bandwidth between nodes using iperf.

        Args:
            src: Source node name
            dst: Destination node name

        Returns:
            Dictionary with bandwidth statistics
        """
        src_node = self.get_node(src)
        dst_node = self.get_node(dst)

        if not src_node or not dst_node:
            raise ValueError("Node not found")

        # Start iperf server on destination
        dst_node.cmd("iperf -s &")

        # Run iperf client on source
        dst_ip = dst_node.IP()
        output = src_node.cmd(f"iperf -c {dst_ip} -t 10")

        logger.info(f"Bandwidth {src} -> {dst}:\n{output}")

        return {
            "src": src,
            "dst": dst,
            "output": output
        }

    def _parse_bandwidth(self, bandwidth: str) -> float:
        """Parse bandwidth string to float (Mbps)"""
        if bandwidth.endswith("Mbps"):
            return float(bandwidth[:-4])
        elif bandwidth.endswith("Gbps"):
            return float(bandwidth[:-4]) * 1000
        elif bandwidth.endswith("Kbps"):
            return float(bandwidth[:-4]) / 1000
        else:
            return float(bandwidth)

    def get_statistics(self) -> Dict:
        """Get network statistics"""
        return {
            "topology": self.config.name if self.config else "None",
            "nodes": len(self.nodes),
            "mininet_available": self.mininet_available,
            "running": self.net is not None
        }


def create_simple_topology() -> TopologyConfig:
    """Create a simple client-server topology"""
    nodes = [
        NodeConfig(name="client", ip="10.0.0.1", role="client"),
        NodeConfig(name="server", ip="10.0.0.2", role="server")
    ]

    links = [
        LinkConfig(
            node1="client",
            node2="server",
            bandwidth="100Mbps",
            delay="10ms",
            loss=0.0
        )
    ]

    return TopologyConfig(
        name="simple",
        nodes=nodes,
        links=links
    )


def create_complex_topology() -> TopologyConfig:
    """Create a complex multi-hop topology"""
    nodes = [
        NodeConfig(name="client1", ip="10.0.0.1", role="client"),
        NodeConfig(name="client2", ip="10.0.0.2", role="client"),
        NodeConfig(name="router1", ip="10.0.0.254", role="router"),
        NodeConfig(name="router2", ip="10.0.1.254", role="router"),
        NodeConfig(name="server1", ip="10.0.1.1", role="server"),
        NodeConfig(name="server2", ip="10.0.1.2", role="server")
    ]

    links = [
        LinkConfig("client1", "router1", "100Mbps", "5ms", 0.0),
        LinkConfig("client2", "router1", "100Mbps", "5ms", 0.0),
        LinkConfig("router1", "router2", "1Gbps", "20ms", 0.1),
        LinkConfig("router2", "server1", "100Mbps", "5ms", 0.0),
        LinkConfig("router2", "server2", "100Mbps", "5ms", 0.0)
    ]

    return TopologyConfig(
        name="complex",
        nodes=nodes,
        links=links
    )
