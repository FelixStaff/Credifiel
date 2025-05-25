import diagramaImg from "../../assets/diagrama.jpg";

export default function Sustento({ mini = false }: { mini?: boolean }) {
const Table = () => (
  <div className="w-full max-w-md mx-auto overflow-x-auto rounded-xl shadow-inner border border-[#1D99D6]/30 bg-[#1F2937]">
    <table className="w-full text-[11px] text-white text-center">
      <thead>
        <tr className="bg-[#1D99D6]/10 text-[#1D99D6] uppercase text-[10px] tracking-wider">
          <th className="px-2 py-1">Clase</th>
          <th className="px-2 py-1">Precisión</th>
          <th className="px-2 py-1">Recall</th>
          <th className="px-2 py-1">F1-score</th>
          <th className="px-2 py-1">Soporte</th>
        </tr>
      </thead>
      <tbody>
        <tr className="odd:bg-[#ffffff09]">
          <td className="py-1 font-medium">No Pagado</td>
          <td>0.97</td>
          <td>1.00</td>
          <td>0.99</td>
          <td>3234</td>
        </tr>
        <tr className="odd:bg-[#ffffff09]">
          <td className="py-1 font-medium">Pagado</td>
          <td>1.00</td>
          <td>0.96</td>
          <td>0.98</td>
          <td>2085</td>
        </tr>
        <tr className="border-t border-[#1D99D6]/50">
          <td className="py-1 font-bold text-[#1D99D6]">Accuracy</td>
          <td colSpan={4} className="italic">0.98 (5319 muestras)</td>
        </tr>
        <tr className="odd:bg-[#ffffff09]">
          <td className="py-1 font-medium">Macro Promedio</td>
          <td>0.99</td>
          <td>0.98</td>
          <td>0.98</td>
          <td>5319</td>
        </tr>
        <tr className="odd:bg-[#ffffff09]">
          <td className="py-1 font-medium">Weighted Promedio</td>
          <td>0.98</td>
          <td>0.98</td>
          <td>0.98</td>
          <td>5319</td>
        </tr>
      </tbody>
    </table>
  </div>
);


  if (mini) {
    return <Table />;
  }

  return (
    <div className="w-full h-full flex flex-col gap-5 bg-gradient-to-br from-[#15243D] via-[#0E4385] to-[#111827] text-white p-6 rounded-xl shadow-lg">
      <h2 className="text-2xl font-bold text-center text-[#1D99D6]">Sustento del Modelo Predictivo</h2>

      <p className="text-sm text-gray-300 text-center max-w-2xl mx-auto leading-relaxed">
        Esta sección presenta las métricas de evaluación del modelo de predicción de pagos. Se utiliza para validar la eficacia y confiabilidad de los resultados generados mediante técnicas avanzadas de aprendizaje automático.
      </p>

      <Table />
  <p className="text-xs text-gray-400 italic mt-2 text-center">
          El modelo utiliza una arquitectura basada en atención para analizar secuencias temporales de intentos de cobro.
        </p>
      <div className="flex flex-col items-center mt-4">
        <img
          src={diagramaImg}
          alt="Diagrama del modelo"
          className="max-w-md w-full rounded-xl border border-[#1D99D6]/30 shadow-md"
        />
      
      </div>
    </div>
  );
}
