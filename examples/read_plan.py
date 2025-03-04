import os
import pydicom


def list_beams(ds: pydicom.Dataset) -> str:
    """Summarizes the RTPLAN beam information in the dataset."""
    lines = [f"{'Beam name':^13s} {'Number':^8s} {'Gantry':^8s} {'SSD (cm)':^11s}"]
    for beam in ds.BeamSequence:
        cp0 = beam.ControlPointSequence[0]
        ssd = float(cp0.SourceToSurfaceDistance / 10)
        lines.append(
            f"{beam.BeamName:^13s} {beam.BeamNumber:8d} {cp0.GantryAngle:8.1f} {ssd:8.1f}"
        )
    return "\n".join(lines)


def read_rtplan(filename: str):
    plan = pydicom.dcmread(filename, force=True)
    #print(list_beams(plan))
    print(plan)


if __name__ == "__main__":
    #rtplan_folder = "/Users/amithkamath/data/EORTC-ICR/test_plan/"
    #rtplan_filename = "1.3.6.1.4.1.25111.0095.20180818.165038.00.3.00.RP.dcm"
    rtplan_folder = ("/Users/amithkamath/data/EORTC-ICR/testing/test_plan_2/")
    rtplan_filename = "rtplan.dcm"
    rtplan_file = os.path.join(rtplan_folder, rtplan_filename)
    read_rtplan(rtplan_file)