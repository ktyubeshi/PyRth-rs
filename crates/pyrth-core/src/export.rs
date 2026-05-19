use std::{
    fs,
    path::{Path, PathBuf},
};

use csv::Writer;

use crate::{error::Result, EvaluationResult};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExportedCsvFiles {
    pub impedance: PathBuf,
    pub imp_deriv: Option<PathBuf>,
    pub time_spec: Option<PathBuf>,
    pub foster: Option<PathBuf>,
    pub cauer: Option<PathBuf>,
    pub diff_struc: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExportedFigureFiles {
    pub impedance: PathBuf,
    pub imp_deriv: Option<PathBuf>,
    pub time_spec: Option<PathBuf>,
    pub foster: Option<PathBuf>,
    pub cauer: Option<PathBuf>,
    pub diff_struc: Option<PathBuf>,
}

pub fn export_csv(
    result: &EvaluationResult,
    output_dir: impl AsRef<Path>,
) -> Result<ExportedCsvFiles> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let impedance_path = output_dir.join("impedance.csv");
    write_pairs(
        &impedance_path,
        ["time", "impedance"],
        result.impedance.time.iter().copied(),
        result.impedance.impedance.iter().copied(),
    )?;

    let mut files = ExportedCsvFiles {
        impedance: impedance_path,
        ..ExportedCsvFiles::default()
    };

    if let Some(derivative) = result.derivative.as_ref() {
        let imp_deriv_path = output_dir.join("imp_deriv.csv");
        write_pairs(
            &imp_deriv_path,
            ["time", "imp_deriv"],
            derivative.log_time_pad.iter().map(|value| value.exp()),
            derivative.imp_deriv_interp.iter().copied(),
        )?;
        files.imp_deriv = Some(imp_deriv_path);

        if let Some(time_spectrum) = result.time_spectrum.as_ref() {
            let time_spec_path = output_dir.join("time_spec.csv");
            write_pairs(
                &time_spec_path,
                ["time", "time_spec"],
                derivative.log_time_pad.iter().map(|value| value.exp()),
                time_spectrum.iter().copied(),
            )?;
            files.time_spec = Some(time_spec_path);
        }
    }

    if let Some(foster) = result.foster.as_ref() {
        let foster_path = output_dir.join("foster.csv");
        write_pairs(
            &foster_path,
            ["resistance", "capacitance"],
            foster.resistance.iter().copied(),
            foster.capacitance.iter().copied(),
        )?;
        files.foster = Some(foster_path);
    }

    if let Some(cauer) = result.cauer.as_ref() {
        let cauer_path = output_dir.join("cauer.csv");
        write_pairs(
            &cauer_path,
            ["cumulative_resistance", "cumulative_capacitance"],
            cauer.cumulative_resistance.iter().copied(),
            cauer.cumulative_capacitance.iter().copied(),
        )?;
        files.cauer = Some(cauer_path);

        let diff_struc_path = output_dir.join("diff_struc.csv");
        write_pairs(
            &diff_struc_path,
            ["cumulative_resistance", "differential_structure"],
            cauer.cumulative_resistance.iter().copied(),
            cauer.differential_structure.iter().copied(),
        )?;
        files.diff_struc = Some(diff_struc_path);
    }

    Ok(files)
}

pub fn export_svg_figures(
    result: &EvaluationResult,
    output_dir: impl AsRef<Path>,
) -> Result<ExportedFigureFiles> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let impedance_path = output_dir.join("impedance.svg");
    write_svg_series(
        &impedance_path,
        "Thermal impedance",
        "time",
        "impedance",
        result.impedance.time.iter().copied(),
        result.impedance.impedance.iter().copied(),
        AxisScale::Log10,
        AxisScale::Linear,
    )?;

    let mut files = ExportedFigureFiles {
        impedance: impedance_path,
        ..ExportedFigureFiles::default()
    };

    if let Some(derivative) = result.derivative.as_ref() {
        let imp_deriv_path = output_dir.join("imp_deriv.svg");
        write_svg_series(
            &imp_deriv_path,
            "Impedance derivative",
            "time",
            "imp_deriv",
            derivative.log_time_pad.iter().map(|value| value.exp()),
            derivative.imp_deriv_interp.iter().copied(),
            AxisScale::Log10,
            AxisScale::Linear,
        )?;
        files.imp_deriv = Some(imp_deriv_path);

        if let Some(time_spectrum) = result.time_spectrum.as_ref() {
            let time_spec_path = output_dir.join("time_spec.svg");
            write_svg_series(
                &time_spec_path,
                "Time constant spectrum",
                "time",
                "time_spec",
                derivative.log_time_pad.iter().map(|value| value.exp()),
                time_spectrum.iter().copied(),
                AxisScale::Log10,
                AxisScale::Linear,
            )?;
            files.time_spec = Some(time_spec_path);
        }
    }

    if let Some(foster) = result.foster.as_ref() {
        let foster_path = output_dir.join("foster.svg");
        write_svg_series(
            &foster_path,
            "Foster network",
            "resistance",
            "capacitance",
            foster.resistance.iter().copied(),
            foster.capacitance.iter().copied(),
            AxisScale::Linear,
            AxisScale::Linear,
        )?;
        files.foster = Some(foster_path);
    }

    if let Some(cauer) = result.cauer.as_ref() {
        let cauer_path = output_dir.join("cauer.svg");
        write_svg_series(
            &cauer_path,
            "Cauer structure",
            "cumulative_resistance",
            "cumulative_capacitance",
            cauer.cumulative_resistance.iter().copied(),
            cauer.cumulative_capacitance.iter().copied(),
            AxisScale::Linear,
            AxisScale::Log10,
        )?;
        files.cauer = Some(cauer_path);

        let diff_struc_path = output_dir.join("diff_struc.svg");
        write_svg_series(
            &diff_struc_path,
            "Differential structure",
            "cumulative_resistance",
            "differential_structure",
            cauer.cumulative_resistance.iter().copied(),
            cauer.differential_structure.iter().copied(),
            AxisScale::Linear,
            AxisScale::Log10,
        )?;
        files.diff_struc = Some(diff_struc_path);
    }

    Ok(files)
}

fn write_pairs(
    path: &Path,
    header: [&str; 2],
    left: impl Iterator<Item = f64>,
    right: impl Iterator<Item = f64>,
) -> Result<()> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(header)?;
    for (left, right) in left.zip(right) {
        writer.write_record([format!("{left:.17e}"), format!("{right:.17e}")])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_svg_series(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    left: impl Iterator<Item = f64>,
    right: impl Iterator<Item = f64>,
    x_scale: AxisScale,
    y_scale: AxisScale,
) -> Result<()> {
    let points = left
        .zip(right)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    let svg = render_svg_series(title, x_label, y_label, &points, x_scale, y_scale);
    fs::write(path, svg)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AxisScale {
    Linear,
    Log10,
}

impl AxisScale {
    fn label(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Log10 => "log10",
        }
    }

    fn transform(self, value: f64) -> Option<f64> {
        match self {
            Self::Linear => Some(value),
            Self::Log10 if value > 0.0 => Some(value.log10()),
            Self::Log10 => None,
        }
    }

    fn inverse(self, value: f64) -> f64 {
        match self {
            Self::Linear => value,
            Self::Log10 => 10.0_f64.powf(value),
        }
    }
}

fn render_svg_series(
    title: &str,
    x_label: &str,
    y_label: &str,
    points: &[(f64, f64)],
    x_scale: AxisScale,
    y_scale: AxisScale,
) -> String {
    const WIDTH: f64 = 720.0;
    const HEIGHT: f64 = 420.0;
    const LEFT: f64 = 72.0;
    const RIGHT: f64 = 24.0;
    const TOP: f64 = 42.0;
    const BOTTOM: f64 = 58.0;

    let plot_width = WIDTH - LEFT - RIGHT;
    let plot_height = HEIGHT - TOP - BOTTOM;
    let points = points
        .iter()
        .filter_map(|(x, y)| Some((x_scale.transform(*x)?, y_scale.transform(*y)?)))
        .collect::<Vec<_>>();
    let (min_x, max_x, min_y, max_y) = bounds(&points);
    let polyline = points
        .iter()
        .map(|(x, y)| {
            let px = LEFT + normalize(*x, min_x, max_x) * plot_width;
            let py = TOP + (1.0 - normalize(*y, min_y, max_y)) * plot_height;
            format!("{px:.3},{py:.3}")
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH:.0} {HEIGHT:.0}" role="img" data-x-scale="{}" data-y-scale="{}">
  <title>{}</title>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{:.0}" y="24" font-family="sans-serif" font-size="18" fill="#111111">{}</text>
  <line x1="{LEFT:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="#222222" stroke-width="1"/>
  <line x1="{LEFT:.0}" y1="{TOP:.0}" x2="{LEFT:.0}" y2="{:.0}" stroke="#222222" stroke-width="1"/>
  <polyline fill="none" stroke="#0f766e" stroke-width="2" points="{}"/>
  <text x="{:.0}" y="{:.0}" font-family="sans-serif" font-size="12" fill="#333333">{}</text>
  <text x="12" y="{:.0}" font-family="sans-serif" font-size="12" fill="#333333" transform="rotate(-90 12,{:.0})">{}</text>
  <text x="{LEFT:.0}" y="{:.0}" font-family="monospace" font-size="11" fill="#555555">{:.3e}</text>
  <text x="{:.0}" y="{:.0}" font-family="monospace" font-size="11" fill="#555555" text-anchor="end">{:.3e}</text>
  <text x="62" y="{:.0}" font-family="monospace" font-size="11" fill="#555555" text-anchor="end">{:.3e}</text>
  <text x="62" y="{TOP:.0}" font-family="monospace" font-size="11" fill="#555555" text-anchor="end">{:.3e}</text>
</svg>
"##,
        x_scale.label(),
        y_scale.label(),
        escape_xml(title),
        LEFT,
        escape_xml(title),
        TOP + plot_height,
        LEFT + plot_width,
        TOP + plot_height,
        TOP + plot_height,
        polyline,
        LEFT + plot_width / 2.0,
        HEIGHT - 16.0,
        escape_xml(x_label),
        TOP + plot_height / 2.0,
        TOP + plot_height / 2.0,
        escape_xml(y_label),
        TOP + plot_height + 18.0,
        x_scale.inverse(min_x),
        LEFT + plot_width,
        TOP + plot_height + 18.0,
        x_scale.inverse(max_x),
        TOP + plot_height,
        y_scale.inverse(min_y),
        y_scale.inverse(max_y),
    )
}

fn bounds(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    if points.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let (mut min_x, mut max_x) = (points[0].0, points[0].0);
    let (mut min_y, mut max_y) = (points[0].1, points[0].1);
    for (x, y) in points.iter().copied() {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if min_x == max_x {
        min_x -= 0.5;
        max_x += 0.5;
    }
    if min_y == max_y {
        min_y -= 0.5;
        max_y += 0.5;
    }
    (min_x, max_x, min_y, max_y)
}

fn normalize(value: f64, min: f64, max: f64) -> f64 {
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
