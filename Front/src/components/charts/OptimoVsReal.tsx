import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import jsonData from "../../data/OptimoVsReal.json";

const months = ["Ene", "Feb", "Mar", "Abr", "May"];

export default function OptimoVsReal({ mini = false }: { mini?: boolean }) {
  const [data, setData] = useState<{ mes: string; real: number; optimo: number }[]>([]);

  useEffect(() => {
    const totalValues = jsonData.length;
    const perMonth = totalValues / 5;
    const parsedData = months.map((mes, i) => {
      const start = Math.floor(i * perMonth);
      const end = Math.floor((i + 1) * perMonth);
      const slice = jsonData.slice(start, end);
      const real = slice.reduce((sum, d) =>  d.original, 0);
      const optimo = slice.reduce((sum, d) => d.optimizado, 0);
      return { mes, real, optimo };
    });
    setData(parsedData);
  }, []);

if (mini) {
  // Tooltip personalizado para mostrar ambos valores
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#1F2937] text-white text-xs p-2 rounded shadow-md">
          <p className="font-bold mb-1">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={`item-${index}`} className="flex items-center space-x-1">
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span>
                {entry.name}:{" "}
                {entry.value >= 1e9
                  ? `$${(entry.value / 1e9).toFixed(0)}B`
                  : entry.value >= 1e6
                  ? `$${(entry.value / 1e6).toFixed(0)}M`
                  : entry.value >= 1e3
                  ? `$${(entry.value / 1e3).toFixed(0)}K`
                  : `$${entry.value.toFixed(0)}`}
              </span>
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full relative">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 10, right: 10, bottom: 30, left: -10 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1D99D680" />
          <XAxis dataKey="mes" stroke="#ffffff" tick={{ fontSize: 10 }} />
          <YAxis
            stroke="#ffffff"
            tick={{ fontSize: 10 }}
            domain={['dataMin - 1000000', 'dataMax + 1000000']}
            tickFormatter={(v) => {
              if (v >= 1e9) return `$${(v / 1e9).toFixed(0)}B`;
              if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
              if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
              return `$${v.toFixed(0)}`;

            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="real"
            stroke="#0E4385"
            dot={true}
            strokeWidth={2}
            name="Ganancia Real"
          />
          <Line
            type="monotone"
            dataKey="optimo"
            stroke="#1D99D6"
            dot={true}
            strokeDasharray="4 2"
            strokeWidth={2}
            name="Ganancia Óptima"
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Leyenda dentro de la gráfica en esquina inferior derecha */}
      <div className="absolute bottom-4 right-6 flex space-x-3 text-xs rounded px-2 py-1 text-white">
        <div className="flex items-center space-x-1">
          <span className="w-2 h-2 rounded-full bg-[#0E4385] inline-block" />
          <span>Ganancia Real</span>
        </div>
        <div className="flex items-center space-x-1">
          <span className="w-2 h-2 rounded-full bg-[#1D99D6] inline-block border-dashed border-2 border-[#1D99D6]" />
          <span>Ganancia Óptima</span>
        </div>
      </div>
    </div>
  );
}


const lastEntry = data[data.length - 1];
const delta = lastEntry ? lastEntry.optimo - lastEntry.real : 0;


  // ✅ Función para formatear montos de forma legible
  const formatCurrency = (v: number) => {
    if (v >= 1e9) return `$${(v / 1e9).toFixed(0)}B`;
    if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
    if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
    return `$${v.toFixed(0)}`;
  };

  return (
    <div className="w-full h-full flex flex-col gap-4 bg-gradient-to-br from-[#15243D] via-[#0E4385] to-[#111827] text-white p-6 rounded-xl shadow-lg">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-extrabold text-white text-center">
          Retorno de Inversión: Óptimo vs Actual
        </h2>
  <p className="absolute left-12 mt-5 text-sm text-gray-300 text-justify ">
    Este gráfico muestra la comparación entre la ganancia acumulada real y la ganancia
    estimada si se hubiera aplicado la estrategia óptima a lo largo del año.
  </p>
      </div>

      {/* Chart */}
      <div className="w-full mt-8 h-[300px] p-4 rounded-lg bg-[#15243D]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1D99D680" />
            <XAxis dataKey="mes" stroke="#ffffff" />
            <YAxis
              stroke="#ffffff"
              tickFormatter={formatCurrency}
              domain={['dataMin - 1000000', 'dataMax + 1000000']}

            />
            <Tooltip
              contentStyle={{ backgroundColor: "#1F2937", border: "none", color: "white" }}
              formatter={(v: number) => formatCurrency(v)}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="real"
              stroke="#FFFFFF"
              name="Ganancia Real"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="optimo"
              stroke="#1D99D6"
              name="Ganancia Óptima"
              strokeDasharray="4 2"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
 {/* Highlight */}
      <div className="text-xl text-gray-300 text-center">
        <strong className="text-white text-2xl">Ganancia adicional potencial:</strong>{" "}
        <span className="text-[#1D99D6] font-semibold text-2xl">
{delta.toLocaleString()}
        </span>
{/* Tabla horizontal de porcentajes por mes */}
<div className="overflow-x-auto rounded-lg shadow-md bg-[#15243D] p-4 mt-6">
  <h3 className="text-sm font-semibold text-white mb-2">Beneficio Acumulado Mensual</h3>
  <table className="w-full text-sm text-center text-gray-300">
    <thead className="text-xs uppercase text-gray-400 border-b border-[#1D99D6]">
      <tr>
        <th className="px-4 py-2">Mes</th>
        <th className="px-4 py-2">Enero</th>
        <th className="px-4 py-2">Febrero</th>
        <th className="px-4 py-2">Marzo</th>
        <th className="px-4 py-2">Abril</th>
        <th className="px-4 py-2">Mayo</th>
      </tr>
    </thead>
    <tbody>
      <tr className="border-t border-gray-600">
        <td className="px-4 py-2 font-semibold text-gray-400">Beneficio</td>
        <td className="px-4 py-2">$342,869</td>
        <td className="px-4 py-2">$670,114</td>
        <td className="px-4 py-2">$998,906</td>
        <td className="px-4 py-2">$1,328,274</td>
        <td className="px-4 py-2">$1,685,099.95</td>
      </tr>
    </tbody>
  </table>
</div>

      </div>
      {/* Insights & Actions */}
     <div className="bg-[#15243D] p-5 rounded-lg shadow-lg border-l-4 border-[#1D99D6]">
  <h3 className="text-sm font-semibold text-white mb-2">Observaciones</h3>
  <ul className="list-disc text-sm text-gray-300 pl-5 space-y-1">
    <li>Se recomienda utilizar Banco Banamex cuando el número de intentos sea menor o igual a 5 y el modelo prediga que el cliente pagará, salvo que el banco del cliente sea Santander, en cuyo caso debe usarse Santander.</li>
    <li>Para intentos mayores a 5 con predicción positiva, o menores a 10 con predicción negativa, se sugiere usar BBVA Interbancario por eficiencia en el procesamiento.</li>
    <li>En casos con 7 a 10 intentos y predicción negativa, Banorte es preferible si coincide con el banco del cliente; en caso contrario, se recomienda Santander.</li>
    <li>Si el número de intentos supera los 20 con predicción positiva, o los 10 con predicción negativa, se recomienda suspender el envío de peticiones para evitar costos innecesarios.</li>
  </ul>
  <button
    className="mt-3 px-5 py-2 bg-[#0E4385] hover:bg-[#1D99D6] transition-colors duration-300 text-white text-sm rounded shadow-md"
    onClick={() => alert("Aplicando acciones recomendadas (modo demo)...")}
  >
    Aplicar solución recomendada
  </button>
</div>

    </div>
  );
}
