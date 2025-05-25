import rawData from "../data/Resultados.json";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const differences = rawData.map((entry: any) => +(entry.optimizado - entry.original).toFixed(2));
const monthLabels = ["Enero", "Febrero", "Marzo", "Abril", "Mayo"];
const chunkSize = Math.ceil(differences.length / monthLabels.length);

const monthlyDiff = monthLabels.map((month, i) => {
  const chunk = differences.slice(i * chunkSize, (i + 1) * chunkSize);
  const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length;
  return {
    month,
    difference: +avg.toFixed(2),
  };
});

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: any;
  label?: string;
}) => {
  if (active && payload && payload.length > 0) {
    return (
      <div className="bg-white text-black text-xs p-2 rounded shadow border">
        <p>{label}</p>
        <p className="font-semibold">
          Beneficio: ${payload[0].value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </p>
      </div>
    );
  }

  return null;
};

export default function RevenueProjection() {
  return (
    <div className="p-6 bg-gradient-to-br from-[#15243D] via-[#0E4385] to-[#111827] text-white rounded-xl shadow-lg">
      <h2 className="text-2xl font-extrabold text-white mb-4">Proyección de Beneficios</h2>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Gráfica de crecimiento</h3>
        <div className="w-full h-64 bg-[#1F2937] border border-[#1D99D6]/30 rounded">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={monthlyDiff} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1D99D680" />
              <XAxis dataKey="month" stroke="#D1D5DB" />
              <YAxis
                stroke="#D1D5DB"
                tickFormatter={(value) =>
                  `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                }
              />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="difference" stroke="#3b82f6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full bg-[#1F2937] text-white text-sm rounded-md border border-[#1D99D6]/30 shadow-inner">
          <thead className="bg-[#0E4385] text-left uppercase text-xs text-white tracking-wider">
            <tr>
              <th className="px-4 py-3">Mes</th>
              <th className="px-4 py-3">Beneficio Promedio</th>
            </tr>
          </thead>
          <tbody>
            {monthlyDiff.map((row, index) => (
              <tr key={index} className="border-t border-[#1D99D6]/20 hover:bg-[#2C3E50]">
                <td className="px-4 py-2">{row.month}</td>
                <td className="px-4 py-2">
                  ${row.difference.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
