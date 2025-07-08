from core.state import ScheduleState

def handle_fixed_assignments(model, state: ScheduleState):
    for (nurse, day_idx), shift_label in state.fixed_assignments.items():
        label = shift_label.strip().upper()

        # Fix MC, REST, AL, EL as no state.work
        if label in {"EL", "MC", "AL", "REST"}:
            # Block all shifts
            for s in range(state.shift_types):
                model.Add(state.work[nurse, day_idx, s] == 0)
            # Record MC overrides
            if label == "MC":
                state.mc_sets[nurse].add(day_idx)
            if label == "AL":
                state.al_sets[nurse].add(day_idx)
            # EL already recorded in el_sets

        # handle double-shifts, e.g. "AM/PM*"
        elif "/" in label:
            # remove any trailing "*" and split
            parts = label.rstrip("*").split("/")
            # validate
            try:
                idxs = [ state.shift_str_to_idx[p] for p in parts ]
            except KeyError as e:
                raise ValueError(f"Unknown shift part '{e.args[0]}' in double-shift '{label}' for {nurse}")
            # force both component shifts on, others off
            for s in idxs:
                model.Add(state.work[nurse, day_idx, s] == 1)
            for other_s in set(range(state.shift_types)) - set(idxs):
                model.Add(state.work[nurse, day_idx, other_s] == 0)

        # Force that one shift and turn off the others
        else:
            if label not in state.shift_str_to_idx:
                raise ValueError(f"Unknown shift '{label}' for {nurse}")
            s = state.shift_str_to_idx[label]
            model.Add(state.work[nurse, day_idx, s] == 1)
            for other_s in (set(range(state.shift_types)) - {s}):
                model.Add(state.work[nurse, day_idx, other_s] == 0)
                