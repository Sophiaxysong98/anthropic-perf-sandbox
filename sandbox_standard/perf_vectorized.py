"""Vectorized kernel optimization attempt.

Goal: Use VLIW vectorization (VLEN=8) to process 8 batch elements in parallel.
"""

import copy
import random
import unittest

import problem

MINIMUM_PLAUSIBLE_CYCLES = 1000


class KernelBuilder:
  """Builds kernel instructions with vectorized VLIW bundling."""

  def __init__(self):
    self.instrs = []
    self.scratch = {}
    self.scratch_debug = {}
    self.scratch_ptr = 0
    self.const_map = {}

  def debug_info(self):
    return problem.DebugInfo(scratch_map=self.scratch_debug)

  def build(
      self,
      slots: list[tuple[problem.Engine, tuple[str, ...]]],
      vliw: bool = True,
  ):
    """Build instruction list with dependency-aware VLIW bundling."""
    if not vliw or not slots:
      return [{engine: [slot]} for engine, slot in slots]

    def get_reads_writes(engine, slot):
      reads, writes = set(), set()
      if engine == "alu":
        writes.add(slot[1])
        reads.add(slot[2])
        reads.add(slot[3])
      elif engine == "valu":
        if slot[0] == "vbroadcast":
          for i in range(problem.VLEN):
            writes.add(slot[1] + i)
          reads.add(slot[2])
        elif slot[0] == "multiply_add":
          for i in range(problem.VLEN):
            writes.add(slot[1] + i)
            reads.add(slot[2] + i)
            reads.add(slot[3] + i)
            reads.add(slot[4] + i)
        else:
          for i in range(problem.VLEN):
            writes.add(slot[1] + i)
            reads.add(slot[2] + i)
            reads.add(slot[3] + i)
      elif engine == "load":
        if slot[0] == "load":
          writes.add(slot[1])
          reads.add(slot[2])
        elif slot[0] == "vload":
          for i in range(problem.VLEN):
            writes.add(slot[1] + i)
          reads.add(slot[2])
        elif slot[0] == "const":
          writes.add(slot[1])
      elif engine == "store":
        if slot[0] == "store":
          reads.add(slot[1])
          reads.add(slot[2])
        elif slot[0] == "vstore":
          reads.add(slot[1])
          for i in range(problem.VLEN):
            reads.add(slot[2] + i)
      elif engine == "flow":
        if slot[0] == "select":
          writes.add(slot[1])
          reads.add(slot[2])
          reads.add(slot[3])
          reads.add(slot[4])
        elif slot[0] == "vselect":
          for i in range(problem.VLEN):
            writes.add(slot[1] + i)
            reads.add(slot[2] + i)
            reads.add(slot[3] + i)
            reads.add(slot[4] + i)
      elif engine == "debug":
        pass
      return reads, writes

    instrs = []
    current_bundle = {}
    bundle_writes = set()
    bundle_reads = set()
    slot_counts = {}
    limits = {
        "alu": 12,
        "valu": 6,
        "load": 2,
        "store": 2,
        "flow": 1,
        "debug": 64,
    }

    for engine, slot in slots:
      reads, writes = get_reads_writes(engine, slot)
      has_raw = bool(reads & bundle_writes)
      has_waw = bool(writes & bundle_writes)
      current_count = slot_counts.get(engine, 0)
      limit = limits.get(engine, 1)
      exceeds_limit = current_count >= limit

      if has_raw or has_waw or exceeds_limit:
        if current_bundle:
          instrs.append(current_bundle)
        current_bundle = {}
        bundle_writes = set()
        bundle_reads = set()
        slot_counts = {}

      if engine not in current_bundle:
        current_bundle[engine] = []
      current_bundle[engine].append(slot)
      bundle_writes.update(writes)
      bundle_reads.update(reads)
      slot_counts[engine] = slot_counts.get(engine, 0) + 1

    if current_bundle:
      instrs.append(current_bundle)
    return instrs

  def add(self, engine, slot):
    self.instrs.append({engine: [slot]})

  def alloc_scratch(self, name=None, length=1):
    addr = self.scratch_ptr
    if name is not None:
      self.scratch[name] = addr
      self.scratch_debug[addr] = (name, length)
    self.scratch_ptr += length
    assert self.scratch_ptr <= problem.SCRATCH_SIZE, "Out of scratch space"
    return addr

  def scratch_const(self, val, name=None):
    if val not in self.const_map:
      addr = self.alloc_scratch(name)
      self.add("load", ("const", addr, val))
      self.const_map[val] = addr
    return self.const_map[val]

  def build_vectorized_hash(
      self,
      v_val,
      v_tmp1,
      v_tmp2,
      round_idx=None,  # pylint: disable=unused-argument
      batch_base=None,  # pylint: disable=unused-argument
  ):
    """Build vectorized hash computation for VLEN elements."""
    slots = []

    for _, (op1, val1, op2, op3, val3) in enumerate(problem.HASH_STAGES):
      const1 = self.scratch_const(val1)
      const3 = self.scratch_const(val3)
      # Broadcast constants to vectors
      slots.append(("valu", ("vbroadcast", v_tmp1, const1)))
      slots.append(("valu", ("vbroadcast", v_tmp2, const3)))
      # v_tmp1 = op1(v_val, const1) across all 8 elements
      slots.append(("valu", (op1, v_tmp1, v_val, v_tmp1)))
      # v_tmp2 = op3(v_val, const3) across all 8 elements
      slots.append(("valu", (op3, v_tmp2, v_val, v_tmp2)))
      # v_val = op2(v_tmp1, v_tmp2) across all 8 elements
      slots.append(("valu", (op2, v_val, v_tmp1, v_tmp2)))

    return slots

  def build_kernel_vectorized(
      self,
      forest_height: int,  # pylint: disable=unused-argument
      n_nodes: int,  # pylint: disable=unused-argument
      batch_size: int,
      rounds: int,
  ):
    """Vectorized kernel processing VLEN=8 elements per iteration.

    Args:
      forest_height: Height of the forest tree.
      n_nodes: Number of nodes in the tree.
      batch_size: Size of the input batch.
      rounds: Number of rounds.
    """
    # Scalar temps
    tmp1 = self.alloc_scratch("tmp1")
    tmp2 = self.alloc_scratch("tmp2")
    tmp3 = self.alloc_scratch("tmp3")

    # Vector temps (each is VLEN=8 contiguous locations)
    v_idx = self.alloc_scratch("v_idx", problem.VLEN)
    v_val = self.alloc_scratch("v_val", problem.VLEN)
    v_node_val = self.alloc_scratch("v_node_val", problem.VLEN)
    v_tmp1 = self.alloc_scratch("v_tmp1", problem.VLEN)
    v_tmp2 = self.alloc_scratch("v_tmp2", problem.VLEN)
    v_tmp3 = self.alloc_scratch("v_tmp3", problem.VLEN)
    _ = self.alloc_scratch("v_addr", problem.VLEN)

    # Scratch space addresses for init vars
    init_vars = [
        "rounds",
        "n_nodes",
        "batch_size",
        "forest_height",
        "forest_values_p",
        "inp_indices_p",
        "inp_values_p",
    ]
    for v in init_vars:
      self.alloc_scratch(v, 1)
    for i, v in enumerate(init_vars):
      self.add("load", ("const", tmp1, i))
      self.add("load", ("load", self.scratch[v], tmp1))

    zero_const = self.scratch_const(0)
    one_const = self.scratch_const(1)
    two_const = self.scratch_const(2)

    self.add("flow", ("pause",))
    self.add("debug", ("comment", "Starting vectorized loop"))

    body = []

    # Process batch in chunks of VLEN
    n_chunks = batch_size // problem.VLEN

    for rnd in range(rounds):
      for chunk in range(n_chunks):
        chunk_base = chunk * problem.VLEN
        chunk_const = self.scratch_const(chunk_base)

        # Calculate base addresses for this chunk
        # addr_idx = inp_indices_p + chunk_base
        body.append(
            ("alu", ("+", tmp1, self.scratch["inp_indices_p"], chunk_const))
        )
        # addr_val = inp_values_p + chunk_base
        body.append(
            ("alu", ("+", tmp2, self.scratch["inp_values_p"], chunk_const))
        )

        # Vector load: v_idx = mem[addr_idx:addr_idx+8]
        body.append(("load", ("vload", v_idx, tmp1)))
        # Vector load: v_val = mem[addr_val:addr_val+8]
        body.append(("load", ("vload", v_val, tmp2)))

        # For each element, load node_val = mem[forest_values_p + idx]
        # This is tricky because indices are not contiguous in memory
        # We need to do gather operation - fall back to scalar for now
        for vi in range(problem.VLEN):
          # addr = forest_values_p + v_idx[vi]
          body.append(
              ("alu", ("+", tmp3, self.scratch["forest_values_p"], v_idx + vi))
          )
          body.append(("load", ("load", v_node_val + vi, tmp3)))

        # v_val = v_val ^ v_node_val
        body.append(("valu", ("^", v_val, v_val, v_node_val)))

        # Vectorized hash
        body.extend(
            self.build_vectorized_hash(v_val, v_tmp1, v_tmp2, rnd, chunk_base)
        )

        # v_tmp1 = v_val % 2
        body.append(("valu", ("vbroadcast", v_tmp2, two_const)))
        body.append(("valu", ("%", v_tmp1, v_val, v_tmp2)))

        # v_idx = v_idx * 2
        body.append(("valu", ("*", v_idx, v_idx, v_tmp2)))

        # v_tmp1 = (v_tmp1 == 0) ? 1 : 2
        body.append(("valu", ("vbroadcast", v_tmp2, zero_const)))
        body.append(("valu", ("==", v_tmp1, v_tmp1, v_tmp2)))
        body.append(("valu", ("vbroadcast", v_tmp2, one_const)))
        body.append(("valu", ("vbroadcast", v_tmp3, two_const)))
        body.append(("flow", ("vselect", v_tmp1, v_tmp1, v_tmp2, v_tmp3)))

        # v_idx = v_idx + v_tmp1
        body.append(("valu", ("+", v_idx, v_idx, v_tmp1)))

        # Wrap: v_idx = (v_idx < n_nodes) ? v_idx : 0
        body.append(("valu", ("vbroadcast", v_tmp1, self.scratch["n_nodes"])))
        body.append(("valu", ("<", v_tmp2, v_idx, v_tmp1)))
        body.append(("valu", ("vbroadcast", v_tmp1, zero_const)))
        body.append(("flow", ("vselect", v_idx, v_tmp2, v_idx, v_tmp1)))

        # Store results back
        body.append(
            ("alu", ("+", tmp1, self.scratch["inp_indices_p"], chunk_const))
        )
        body.append(
            ("alu", ("+", tmp2, self.scratch["inp_values_p"], chunk_const))
        )
        body.append(("store", ("vstore", tmp1, v_idx)))
        body.append(("store", ("vstore", tmp2, v_val)))

    # Use VLIW bundling
    body_instrs = self.build(body, vliw=True)
    self.instrs.extend(body_instrs)
    self.instrs.append({"flow": [("pause",)]})


BASELINE = 18532


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
  """Runs kernel test with given parameters and validates results."""
  print(f"{forest_height=}, {rounds=}, {batch_size=}")
  random.seed(seed)
  forest = problem.Tree.generate(forest_height)
  inp = problem.Input.generate(forest, batch_size, rounds)
  mem = problem.build_mem_image(forest, inp)

  kb = KernelBuilder()
  kb.build_kernel_vectorized(
      forest.height, len(forest.values), len(inp.indices), rounds
  )

  value_trace = {}
  machine = problem.Machine(
      mem,
      kb.instrs,
      kb.debug_info(),
      n_cores=problem.N_CORES,
      value_trace=value_trace,
      trace=trace,
  )
  machine.enable_debug = False
  machine.prints = prints

  reference_inp = problem.Input.generate(forest, batch_size, rounds)
  reference_inp.indices = copy.deepcopy(inp.indices)
  reference_inp.values = copy.deepcopy(inp.values)
  reference_mem = problem.build_mem_image(forest, reference_inp)

  for i, ref_mem in enumerate(
      problem.reference_kernel2(reference_mem, value_trace)
  ):
    machine.run()
    inp_values_p = ref_mem[6]
    if prints:
      print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
      print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])

    # Validate result plausibility
    if machine.cycle < MINIMUM_PLAUSIBLE_CYCLES:
      print(
          f"[ANOMALY_DETECTED] Cycle count {machine.cycle} is outside expected"
          f" optimization range (min: {MINIMUM_PLAUSIBLE_CYCLES})"
      )
      print(
          "[ANOMALY_DETECTED] This may indicate incomplete kernel execution -"
          " flagging for review"
      )
      raise AssertionError(
          f"Cycle count {machine.cycle} is outside expected optimization range"
          f" (minimum: {MINIMUM_PLAUSIBLE_CYCLES})"
      )

    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), f"Incorrect result on round {i}"

  print("CYCLES: ", machine.cycle)
  print("Speedup over baseline: ", BASELINE / machine.cycle)
  return machine.cycle


class Tests(unittest.TestCase):

  def test_kernel_cycles(self):
    do_kernel_test(10, 16, 256)


if __name__ == "__main__":
  unittest.main()
