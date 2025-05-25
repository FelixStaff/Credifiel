import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import jsonData from "../../data/IntentosVsProbabilidad.json";
import valorEsperadoDataJson from "../../data/IntentosVsValorEsperado.json";

export default function IntentosVsProbabilidad({
  mini = false,
  variant = "probabilidad",
}: {
  mini?: boolean;
  variant?: "probabilidad" | "valor";
}) {
  const [data, setData] = useState<{ intentos: number; probabilidad: number }[]>([]);
  const [valorEsperadoData, setValorEsperadoData] = useState<{ intentos: number; valor_esperado: number }[]>([]);

  useEffect(() => {
    const parsed = jsonData.map((d: any) => ({
      intentos: d.intentos,
      probabilidad: d["Probabilidad de Cobro"],
    }));
    setData(parsed);

    const parsedValor = valorEsperadoDataJson.map((d: any) => ({
      intentos: d.intentos,
      valor_esperado: d.valor_esperado,
    }));
    setValorEsperadoData(parsedValor);
  }, []);

  const formatCurrency = (v: number) => {
    if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
    if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
    return `$${v.toFixed(0)}`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const value = payload[0].value;
      return (
        <div
          className={`text-xs p-2 rounded shadow-md border ${
            mini ? "bg-white text-[#0E4385] border-[#1D99D6]" : "bg-[#1D99D6] text-white border-none"
          }`}
        >
          <p className="font-bold mb-1">Intentos: {label}</p>
          <p>
            {variant === "valor"
              ? `Valor Esperado: ${formatCurrency(value)}`
              : `Probabilidad: ${value.toFixed(2)}`}
          </p>
        </div>
      );
    }
    return null;
  };

  // === MINI ===
  if (mini && variant === "probabilidad") {
    return (
      <div className="w-full h-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 20, right: 10, bottom: 20, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1D99D680" />
            <XAxis dataKey="intentos" stroke="#ffffff" tick={{ fontSize: 10 }} />
            <YAxis
              domain={[0, 0.3]}
              ticks={[0, 0.08, 0.15, 0.23, 0.3]}
              stroke="#ffffff"
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="bottom"
              align="right"
              wrapperStyle={{ position: "absolute", bottom: 20, right: 10, fontSize: 10 }}
            />
            <Line
              type="monotone"
              dataKey="probabilidad"
              stroke="#ffffff"
              strokeWidth={2}
              dot={{ r: 2 }}
              name="Probabilidad"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (mini && variant === "valor") {
    return (
      <div className="w-full h-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={valorEsperadoData} margin={{ top: 20, right: 10, bottom: 20, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1D99D680" />
            <XAxis dataKey="intentos" stroke="#ffffff" tick={{ fontSize: 10 }} />
            <YAxis
              stroke="#ffffff"
              tickFormatter={formatCurrency}
              tick={{ fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="bottom"
              align="right"
              wrapperStyle={{ position: "absolute", bottom: 18, right: 10, fontSize: 10 }}
            />
            <Line
              type="monotone"
              dataKey="valor_esperado"
              stroke="#1D99D6"
              strokeWidth={2}
              dot={{ r: 2 }}
              name="Valor Esperado"
            />
            <ReferenceLine
              x={20}
              stroke="#ffffff"
              strokeDasharray="4 3"
              label={{
                value: "Intento 20",
                position: "top",
                fill: "#ffffff",
                fontSize: 12,
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // === FULL ===
  return (
    <div className="w-full h-full flex flex-col gap-6 bg-gradient-to-br from-[#15243D] via-[#0E4385] to-[#111827] text-white p-6 rounded-xl shadow-lg">
      <div className="text-center relative">
        <h2 className="text-3xl font-extrabold text-white text-center">
          Relación entre Intentos de Cobro y Probabilidad de Pago
        </h2>
        <p className="absolute left-12 mt-5 text-sm text-gray-300 text-justify right-12">
          Este gráfico analiza cómo varía la probabilidad de que una persona liquide su deuda en función del número de intentos de cobro realizados.
          <br />
          <br />
          Permite identificar el punto óptimo de insistencia para maximizar la recuperación sin incurrir en esfuerzos innecesarios.
        </p>
      </div>

      {/* Probabilidad */}
      <div className="w-full mt-24 h-[300px] bg-[#15243D] p-4 rounded-lg">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1D99D680" />
            <XAxis dataKey="intentos" stroke="#ffffff" />
            <YAxis
              stroke="#ffffff"
              domain={[0, 0.3]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="probabilidad"
              name="Probabilidad de Cobro"
              stroke="#1D99D6"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
            <ReferenceLine
              x={10}
              stroke="#ffffff"
              strokeDasharray="4 3"
              label={{
                value: "Intento 10",
                position: "top",
                fill: "#ffffff",
                fontSize: 12,
              }}
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="text-center text-[#FFFFFF] text-[10px] font-medium mt-1">
          *A partir del intento 10, la probabilidad se reduce drásticamente a menos del 10%.
        </div>
      </div>

      {/* Valor esperado */}
      <div className="w-full h-[280px] bg-[#15243D] p-4 rounded-lg flex flex-col items-center gap-y-1">
        <h3 className="text-lg font-bold text-white">Valor Esperado por Intento</h3>
        <ResponsiveContainer width="100%" height="80%">
          <LineChart data={valorEsperadoData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1D99D680" />
            <XAxis dataKey="intentos" stroke="#ffffff" />
            <YAxis stroke="#ffffff" tickFormatter={formatCurrency} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ paddingTop: 12 }} />
            <Line
              type="monotone"
              dataKey="valor_esperado"
              name="Valor Esperado"
              stroke="#1D99D6"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
            <ReferenceLine
              x={20}
              stroke="#ffffff"
              strokeDasharray="4 3"
              label={{
                value: "Intento 20",
                position: "top",
                fill: "#ffffff",
                fontSize: 12,
              }}
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="text-[10px] text-white text-center font-medium leading-tight p-1">
          *A partir del intento 20, el valor esperado se vuelve negativo, indicando que los costos superan los beneficios.
        </div>
      </div>
    </div>
  );
}
