import { useState } from "react";
import { Maximize2, Minimize2 } from "lucide-react";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import Sustento from "./charts/Sustento";
import Cobranza from "./charts/Cobranza";
import OptimoVsReal from "./charts/OptimoVsReal";

const availableModules = [
  { id: "optimo", label: "Óptimo vs Real", component: <OptimoVsReal mini /> },
  { id: "sustento", label: "Sustento", component: <Sustento mini /> },
  { id: "cobranza_prob", label: "Cobranza - Probabilidad", component: <Cobranza mini variant="probabilidad" /> },
  { id: "cobranza_valor", label: "Cobranza - Valor Esperado", component: <Cobranza mini variant="valor" /> },
  {
    id: "matriz",
    label: "Matriz de Confusión",
    component: (
      <div className="w-full  flex flex-col justify-center items-center bg-[#15243D] text-white p-4 rounded-xl shadow">
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div></div>
          <div className="text-center text-[#D1D5DB]">Predicho: Negativo</div>
          <div className="text-center text-[#D1D5DB]">Predicho: Positivo</div>

          <div className="text-[#D1D5DB] flex items-center">Real: Negativo</div>
          <div className="bg-[#0E4385] p-2 text-center rounded">3234</div>
          <div className="bg-[#0E4385] p-2 text-center rounded">0</div>

          <div className="text-[#D1D5DB] flex items-center">Real: Positivo</div>
          <div className="bg-[#0E4385] p-2 text-center rounded">86</div>
          <div className="bg-[#0E4385] p-2 text-center rounded">1999</div>
        </div>
      </div>
    ),
  },
];

const availableKpis = [
  {
    id: "kpi_negativo",
    label: "Valor Esperado Negativo",
    component: (
      <>
        <span className="text-sm">
          A partir del intento de cobro <span className="font-semibold text-[#1D99D6]">20</span>, el valor esperado es{" "}
<span className="font-semibold text-[#F87171]">negativo</span>
        </span>
      </>
    ),
  },
  {
    id: "kpi_ganancia",
    label: "Ganancia Potencial",
    component: (
      <div className="flex flex-col items-center">
<span className="text-sm text-[#D1D5DB]">Ganancia potencial adicional</span>
        <span className="text-2xl font-bold text-[#1D99D6] mt-1">$1,685,099.95</span>
      </div>
    ),
  },

];

export default function HomePanel() {
  const [editMode, setEditMode] = useState(false);
  const [activeModules, setActiveModules] = useState<string[]>([
    "optimo",
    "sustento",
    "cobranza_prob",
    "cobranza_valor",
    "matriz",
    "kpi_negativo",
    "kpi_ganancia",
    "kpi_boton",
  ]);
  const [wideModules, setWideModules] = useState<string[]>([]);

  const toggleModule = (id: string) => {
    setActiveModules((prev) =>
      prev.includes(id) ? prev.filter((mod) => mod !== id) : [...prev, id]
    );
  };

  const toggleWide = (id: string) => {
    setWideModules((prev) =>
      prev.includes(id) ? prev.filter((mod) => mod !== id) : [...prev, id]
    );
  };

  const visibleKpis = availableKpis.filter((k) => activeModules.includes(k.id));
  const visibleModules = availableModules.filter((m) => activeModules.includes(m.id));

const exportDashboardToPDF = async () => {
  const pdf = new jsPDF("p", "pt", "a4");
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  let yOffset = 50;
  let count = 1;

  pdf.setFont("Helvetica", "bold");
  pdf.setFontSize(26);
  pdf.setTextColor("#1D99D6");
  pdf.text("Reporte de Dashboard", pageWidth / 2, yOffset, { align: "center" });
  yOffset += 40;

  const descriptions: Record<string, { title: string; text: string }> = {
    "kpi_negativo": {
      title: "Indicador: Valor Esperado Negativo",
      text: "A partir del intento 20, el valor esperado de recuperación se vuelve negativo, indicando que seguir intentando podría generar más costos que beneficios.",
    },
    "kpi_ganancia": {
      title: "Indicador: Ganancia Potencial",
      text: "Este KPI refleja la diferencia entre la ganancia obtenida actualmente y la ganancia proyectada si se aplicara una estrategia óptima.",
    },
    "optimo": {
      title: "Óptimo vs Real",
      text: "Comparación entre la ganancia acumulada real y la proyectada si se hubiera aplicado la estrategia óptima desde el inicio.",
    },
    "sustento": {
      title: "Sustento de Ciencia de Datos",
      text: "Este módulo documenta las métricas clave del modelo predictivo como precisión, F1-score y exactitud. Sirve como respaldo científico sobre la validez del modelo aplicado.",
    },
    "cobranza_prob": {
      title: "Cobranza - Probabilidad",
      text: "Muestra cómo la probabilidad de que un cliente pague cambia en función del número de intentos de cobranza.",
    },
    "cobranza_valor": {
      title: "Cobranza - Valor Esperado",
      text: "Indica el valor económico promedio que se puede recuperar en función del número de intentos realizados.",
    },
    "matriz": {
      title: "Matriz de Confusión",
      text: "Resumen visual de los aciertos y errores del modelo al predecir pagos: verdadero negativo, falso positivo, falso negativo y verdadero positivo.",
    },
  };

  const elements = document.querySelectorAll(".exportable-graph");

  for (const element of elements) {
    const elementId = (element as HTMLElement).getAttribute("data-id");

    // Recuperar descripción
    const desc = descriptions[elementId ?? ""] ?? null;
    if (desc) {
      pdf.setFontSize(14);
      pdf.setTextColor("#000000");
      pdf.setFont("Helvetica", "bold");
      pdf.text(`${count}. ${desc.title}`, 40, yOffset);
      yOffset += 16;

      pdf.setFont("Helvetica", "normal");
      pdf.setFontSize(10);
      const splitText = pdf.splitTextToSize(desc.text, pageWidth - 80);
      pdf.text(splitText, 40, yOffset);
      yOffset += splitText.length * 12 + 10;
      count++;
    }

    // Renderizar imagen solo si NO es el botón (ese no se visualiza bien)
    if (elementId !== "kpi_boton") {
      const canvas = await html2canvas(element as HTMLElement, {
        scale: 2,
        backgroundColor: "#ffffff",
      });

      const imgData = canvas.toDataURL("image/png");
      const imgProps = pdf.getImageProperties(imgData);
      const maxWidth = pageWidth - 80;
      const scaledHeight = (imgProps.height * maxWidth) / imgProps.width;

      if (yOffset + scaledHeight + 20 > pageHeight) {
        pdf.addPage();
        yOffset = 50;
      }

      const xCentered = (pageWidth - maxWidth) / 2;
      pdf.addImage(imgData, "PNG", xCentered, yOffset, maxWidth, scaledHeight);
      yOffset += scaledHeight + 30;
    }
  }

  pdf.save("dashboard.pdf");
};


  return (
    <div className="w-full p-6 bg-gradient-to-br from-[#15243D] via-[#0E4385] to-[#111827] text-white min-h-screen">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-extrabold text-white text-center">Resumen General</h1>
        <div className="flex gap-3">
          <button
            onClick={exportDashboardToPDF}
            className="text-xs bg-[#0E4385] hover:bg-[#1D99D6] text-white px-4 py-1 rounded shadow"
          >
            Descargar PDF
          </button>
          <button
            onClick={() => setEditMode(!editMode)}
            className="text-xs bg-[#1F2937] hover:bg-[#1D99D6] text-white px-4 py-1 rounded shadow"
          >
            {editMode ? "Finalizar edición" : "Editar tablero"}
          </button>
        </div>
      </div>

      {editMode && (
        <div className="mb-6 p-4 bg-[#15243D] rounded-xl shadow border-l-4 border-[#1D99D6]">
          <h2 className="text-sm font-semibold text-white mb-2">Selecciona los módulos a mostrar</h2>
          <div className="grid grid-cols-2 md:grid-cols-2 gap-3 text-sm text-gray-300">
            {[...availableModules, ...availableKpis].map((mod) => (
              <label key={mod.id} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={activeModules.includes(mod.id)}
                  onChange={() => toggleModule(mod.id)}
                  className="accent-[#1D99D6]"
                />
                <span>{mod.label}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* KPIs */}
      {visibleKpis.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {visibleKpis.map((kpi) => (
            <div
              key={kpi.id}
              data-id={kpi.id}
              className="exportable-graph bg-[#15243D] border border-[#1D99D6] rounded-xl shadow p-4 text-white flex items-center justify-center text-center"
            >
              {kpi.component}
            </div>
          ))}
        </div>
      )}

      {/* Módulos */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {visibleModules.map((mod) => (
          <div
            key={mod.id}
            data-id={mod.id}
            className={`exportable-graph bg-[#15243D] border border-[#1D99D6] rounded-xl shadow-lg p-4 h-[220px] relative transition-all ${
              wideModules.includes(mod.id) ? "col-span-full md:col-span-2 w-full" : ""
            }`}
          >
            <div className="flex justify-between items-start mb-2">
              <h2 className="text-sm font-semibold text-white">{mod.label}</h2>
              <button
                onClick={() => toggleWide(mod.id)}
                className="z-20 p-1 rounded-full hover:bg-[#1D99D6]/20 text-white"
              >
                {wideModules.includes(mod.id) ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
              </button>
            </div>
            <div className="w-full h-full">{mod.component}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
