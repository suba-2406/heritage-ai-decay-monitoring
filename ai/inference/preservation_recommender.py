#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Preservation Recommendation & Risk Assessment Engine
Implements automated risk stratification and conservation recommendations according to:
- Senmozhi Research Proposal (Work II, Steps 4, 5, 6)
- Archaeological Survey of India (ASI) Conservation Manual
- ICOMOS Stone Deterioration & Conservation Guidelines

Computes:
1. Defect coverage percentage across monument surface.
2. Dominant defect identification and multi-defect interaction analysis.
3. Structural risk classification: LOW, MEDIUM, or HIGH.
4. Comprehensive, actionable preservation prescriptions with specific materials,
   execution protocols, and monitoring timelines.
5. Exportable conservation reports (JSON & Markdown).
"""

import os
import json
from datetime import datetime

RISK_LEVELS = {
    "LOW": {
        "level": "Low",
        "severity_score": 1,
        "color_hex": "#22c55e",
        "description": "Superficial weathering or undamaged baseline stone. No immediate structural danger.",
        "timeline": "Periodic 12-month monitoring cycle."
    },
    "MEDIUM": {
        "level": "Medium",
        "severity_score": 2,
        "color_hex": "#eab308",
        "description": "Biological colonization or localized dampness. Material loss imminent if left untreated.",
        "timeline": "Conservation intervention recommended within 3 to 6 months."
    },
    "HIGH": {
        "level": "High",
        "severity_score": 3,
        "color_hex": "#ef4444",
        "description": "Structural fractures or active moisture seepage. Severe risk of stone exfoliation and core masonry failure.",
        "timeline": "URGENT intervention required within 15 to 30 days."
    }
}

CONSERVATION_PROTOCOLS = {
    "Crack": {
        "urgency": "High",
        "summary": "Structural Fracture & Fissure Intervention",
        "diagnostic_steps": [
            "Conduct non-destructive Ultrasonic Pulse Velocity (UPV) or Ground Penetrating Radar (GPR) to assess internal fracture depth and void extent.",
            "Install calibrated polycarbonate optical crack monitoring tell-tales across fissure lips to measure active seasonal or seismic displacement.",
            "Inspect foundation perimeter and adjacent lintel beams for differential settlement."
        ],
        "treatment_steps": [
            "Clean fracture channel using low-pressure oil-free compressed air (1.5-2.0 bar) and demineralized water flush.",
            "Gravity or low-pressure injection of breathable, salt-free Natural Hydraulic Lime grout (NHL 2.0 or NHL 3.5) with fine silica sand (1:2 ratio). Strictly avoid Ordinary Portland Cement.",
            "For deep shear fissures (>10mm width), insert Austenitic Grade 316 stainless steel or basalt rebar stitching ties anchored in lime-pozzolana matrix.",
            "Surface finish with sacrificial stone dust and lime pointing to match original granite/sandstone texture and hue."
        ],
        "contraindications": [
            "DO NOT use Portland cement mortars or epoxy resin injections, which create impermeable hard spots causing accelerated spalling of surrounding ancient stone.",
            "DO NOT seal cracks without first determining if active water drainage is discharging through the fissure."
        ]
    },
    "Moss": {
        "urgency": "Medium",
        "summary": "Biological Colonization & Lichen Remediation",
        "diagnostic_steps": [
            "Identify biological genus (green algae, crustose lichen, or bryophyte moss) and examine stone substrate friability beneath colonies.",
            "Measure relative surface moisture using a non-destructive dielectric pinless moisture meter.",
            "Survey proximate micro-climatic factors including tree canopy shade, defective gargoyles, and stagnant splash zones."
        ],
        "treatment_steps": [
            "Pre-wet affected stone with deionized water; gently remove heavy moss cushions using stiff natural tampico or horsehair bristle brushes. Strictly avoid wire brushes.",
            "Apply a low-toxicity, biodegradable biocide poultice (such as quaternary ammonium compound 2-3% w/v or zinc fluorosilicate) with a 24-48 hour dwell time.",
            "Rinse thoroughly with low-pressure nebulous water mist (<3 bar) to clear dead organic residues.",
            "Prune overhanging vegetation to maintain minimum 2-meter air clearance and improve natural solar drying."
        ],
        "contraindications": [
            "DO NOT use high-pressure power washing (>15 bar) which scours the protective ancient quarry patina.",
            "DO NOT use sodium hypochlorite (household bleach) or acidic chemical washes, which introduce destructive soluble chloride salts."
        ]
    },
    "Seepage": {
        "urgency": "High",
        "summary": "Moisture Infiltration & Salt Efflorescence Remediation",
        "diagnostic_steps": [
            "Map moisture gradients across the monument elevation to trace the entry pathway (rising damp, capillary suction, or roof terrace failure).",
            "Sample efflorescence crystals for laboratory ion chromatography to identify damaging sub-surface salts (thenardite, halite, gypsum).",
            "Inspect parapet copings, rainwater spouts, and lead flashings for fractures or blockages."
        ],
        "treatment_steps": [
            "Desalinate active crystallization zones using sacrificial demineralized water and arbocel/sepiolite clay compress poultices (repeated until salt extract conductivity < 200 uS/cm).",
            "Clear and re-point stone plinth joints with hydraulic lime and crushed brick dust (surkhi) breathable mortar.",
            "Repair or re-grade terrace drainage channels to ensure a minimum 1:50 fall away from historic masonry cores.",
            "Once internal stone moisture falls below 5%, apply a breathable oligomeric alkyltrialkoxysilane water-repellent impregnator to repel driving rain while permitting water vapor escape."
        ],
        "contraindications": [
            "DO NOT coat damp stone with impermeable polymeric membranes, acrylic sealers, or oil-based paints, which trap moisture and trigger catastrophic stone spalling.",
            "DO NOT apply silicone surface treatments while active salt efflorescence is occurring."
        ]
    },
    "Normal": {
        "urgency": "Low",
        "summary": "Preventative Maintenance & Continuous Monitoring",
        "diagnostic_steps": [
            "Record baseline high-resolution photographic documentation and multispectral reflectance values.",
            "Log ambient relative humidity, surface temperature, and particulate deposition rates."
        ],
        "treatment_steps": [
            "Perform periodic dry dusting of horizontal moldings and ornamental carvings using soft goat hair brushes.",
            "Inspect stone joints and drainage gargoyles before and after monsoon seasons.",
            "Maintain a 1-meter clear gravel apron around monument plinths to prevent soil splashback."
        ],
        "contraindications": [
            "No chemical or mechanical intervention required. Preserve natural patina."
        ]
    }
}

def calculate_defect_coverage(bounding_boxes, image_width=320, image_height=320):
    """
    Calculates the percentage of image area covered by defect bounding boxes.
    Uses pixel mask union to accurately prevent double-counting of overlapping boxes.
    """
    if not bounding_boxes:
        return 0.0

    mask = bytearray(image_width * image_height)
    
    for box in bounding_boxes:
        # box: [xmin, ymin, xmax, ymax]
        x1 = max(0, min(image_width, int(round(box[0]))))
        y1 = max(0, min(image_height, int(round(box[1]))))
        x2 = max(0, min(image_width, int(round(box[2]))))
        y2 = max(0, min(image_height, int(round(box[3]))))
        
        for y in range(y1, y2):
            row_offset = y * image_width
            for x in range(x1, x2):
                mask[row_offset + x] = 1

    covered_pixels = sum(mask)
    total_pixels = image_width * image_height
    coverage_pct = (covered_pixels / max(1, total_pixels)) * 100.0
    return round(coverage_pct, 2)

def evaluate_monument_risk(detections, image_width=320, image_height=320):
    """
    Evaluates monument risk level based on detected defects and surface coverage.
    detections: list of dicts, each with keys: 'class_name', 'box', 'confidence'
    """
    defect_boxes = [d["box"] for d in detections if d["class_name"] != "Normal"]
    coverage_pct = calculate_defect_coverage(defect_boxes, image_width, image_height)

    class_counts = {}
    class_max_confs = {}
    for d in detections:
        c = d["class_name"]
        class_counts[c] = class_counts.get(c, 0) + 1
        class_max_confs[c] = max(class_max_confs.get(c, 0.0), float(d.get("confidence", 0.0)))

    has_crack = "Crack" in class_counts and class_counts["Crack"] > 0
    has_seepage = "Seepage" in class_counts and class_counts["Seepage"] > 0
    has_moss = "Moss" in class_counts and class_counts["Moss"] > 0

    # Risk Categorization Logic (Senmozhi Work II, Step 6)
    if has_crack or (has_seepage and coverage_pct > 15.0) or coverage_pct > 25.0:
        assigned_risk = "HIGH"
        risk_rationale = (
            f"HIGH RISK triggered: " +
            (f"Structural Crack detected ({class_counts.get('Crack', 0)} fissures). " if has_crack else "") +
            (f"Extensive Seepage detected (Coverage: {coverage_pct}%). " if (has_seepage and coverage_pct > 15.0) else "") +
            (f"Severe aggregate defect coverage ({coverage_pct}%). " if coverage_pct > 25.0 else "")
        )
    elif has_moss or has_seepage or coverage_pct >= 3.0:
        assigned_risk = "MEDIUM"
        risk_rationale = (
            f"MEDIUM RISK triggered: " +
            (f"Biological growth (Moss: {class_counts.get('Moss', 0)} colonies). " if has_moss else "") +
            (f"Localized moisture seepage detected ({class_counts.get('Seepage', 0)} patches). " if has_seepage else "") +
            (f"Surface coverage is {coverage_pct}%. " if coverage_pct >= 3.0 else "")
        )
    else:
        assigned_risk = "LOW"
        risk_rationale = (
            f"LOW RISK: Minimal defect manifestations detected (Coverage: {coverage_pct}%). "
            f"Baseline stone condition stable."
        )

    risk_info = RISK_LEVELS[assigned_risk]

    return {
        "risk_level": assigned_risk,
        "severity_score": risk_info["severity_score"],
        "color_hex": risk_info["color_hex"],
        "description": risk_info["description"],
        "action_timeline": risk_info["timeline"],
        "coverage_percentage": coverage_pct,
        "defect_counts": class_counts,
        "defect_max_confidences": {k: round(v, 4) for k, v in class_max_confs.items()},
        "risk_rationale": risk_rationale.strip()
    }

def generate_preservation_plan(image_name, detections, image_width=320, image_height=320):
    """
    Generates a full preservation and conservation plan for an analyzed monument image.
    """
    risk_assessment = evaluate_monument_risk(detections, image_width, image_height)
    detected_classes = list(risk_assessment["defect_counts"].keys())
    
    if not detected_classes or (len(detected_classes) == 1 and "Normal" in detected_classes):
        target_protocols = ["Normal"]
    else:
        # Prioritize Crack > Seepage > Moss
        priority_order = ["Crack", "Seepage", "Moss"]
        target_protocols = [c for c in priority_order if c in detected_classes]

    action_protocols = []
    for cls in target_protocols:
        if cls in CONSERVATION_PROTOCOLS:
            proto = CONSERVATION_PROTOCOLS[cls].copy()
            proto["defect_type"] = cls
            proto["count"] = risk_assessment["defect_counts"].get(cls, 0)
            proto["max_confidence"] = risk_assessment["defect_max_confidences"].get(cls, 0.0)
            action_protocols.append(proto)

    plan = {
        "report_id": f"REP-DECAY-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "analyzed_image": os.path.basename(image_name),
        "risk_assessment": risk_assessment,
        "total_detections": len(detections),
        "recommended_protocols": action_protocols,
        "preservation_summary": {
            "monument_condition": risk_assessment["risk_level"],
            "primary_concern": target_protocols[0] if target_protocols else "None",
            "recommended_action_summary": (
                action_protocols[0]["summary"] if action_protocols else "Routine maintenance and documentation."
            ),
            "supervising_authority": "Archaeological Survey of India (ASI) / State Archaeology Dept"
        }
    }
    return plan

def format_markdown_report(plan):
    """
    Converts a preservation plan dictionary into a formatted Markdown report.
    """
    ra = plan["risk_assessment"]
    lines = [
        f"# MONUMENT DECAY & PRESERVATION RECOMMENDATION REPORT",
        f"**Report ID:** `{plan['report_id']}`  ",
        f"**Timestamp:** `{plan['timestamp']}`  ",
        f"**Analyzed Image:** `{plan['analyzed_image']}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Structural Risk",
        "",
        f"- **Assigned Risk Level:** **{ra['risk_level']}** ({ra['color_hex']})",
        f"- **Defect Surface Coverage:** **{ra['coverage_percentage']}%**",
        f"- **Urgency Timeline:** {ra['action_timeline']}",
        f"- **Condition Assessment:** {ra['description']}",
        f"- **Diagnostic Rationale:** {ra['risk_rationale']}",
        "",
        "### Detected Defects Breakdown",
        "",
        "| Defect Class | Count | Max Model Confidence | Severity Default |",
        "| :--- | :---: | :---: | :--- |"
    ]

    for c, cnt in ra["defect_counts"].items():
        conf = ra["defect_max_confidences"].get(c, 0.0)
        sev = "High" if c in ["Crack", "Seepage"] else ("Medium" if c == "Moss" else "Low")
        lines.append(f"| **{c}** | {cnt} | {conf:.2f} | {sev} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Recommended Conservation Protocols",
        ""
    ])

    for idx, proto in enumerate(plan["recommended_protocols"], start=1):
        lines.extend([
            f"### Protocol {idx}: {proto['summary']} ({proto['defect_type']})",
            f"**Urgency Level:** {proto['urgency']} | **Identified Instances:** {proto['count']}",
            "",
            "#### A. Diagnostic & Investigation Procedures:",
        ])
        for step in proto["diagnostic_steps"]:
            lines.append(f"- {step}")

        lines.extend([
            "",
            "#### B. Conservation & Treatment Steps:",
        ])
        for step in proto["treatment_steps"]:
            lines.append(f"- {step}")

        lines.extend([
            "",
            "#### C. Contraindications & Prohibited Actions:",
        ])
        for contra in proto["contraindications"]:
            lines.append(f"- **WARNING:** {contra}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Report generated automatically by Heritage AI Decay Monitoring System (Senmozhi Work II).*"
    ])

    return "\n".join(lines)

if __name__ == "__main__":
    # Test sample execution
    sample_dets = [
        {"class_name": "Crack", "box": [50, 50, 150, 100], "confidence": 0.38},
        {"class_name": "Seepage", "box": [100, 120, 280, 260], "confidence": 0.44}
    ]
    p = generate_preservation_plan("test_temple_image.jpg", sample_dets, 320, 320)
    print("Risk Assessment:", p["risk_assessment"])
    print("\n--- SAMPLE REPORT PREVIEW ---")
    print(format_markdown_report(p)[:700] + "...")
