/* eslint-disable @typescript-eslint/no-unused-vars */
import { useEffect, useState, useRef } from "react";
import clusterData from "../../data/clusters.json";

interface ClusterPoint {
  id: string;
  cluster: string;
  value: number;
  color: string;
  type: string;
  status: string;
  topPercent: number;
  leftPercent: number;
  propor: number;
}

interface ClusterCenter {
  leftPercent: number;
  topPercent: number;
  ponderaciones: Record<string, number>;
}

interface ClusterPanelProps {
  mini?: boolean;
}

const logoMapping: Record<string, { src: string; keys: string[]; color: string }> = {
  banamex: {
    src: "/src/assets/banamex.png",
    keys: ["BANAMEX CLABE TRADICIONAL"],
    color: "#16a34a",
  },
  banorte: {
    src: "/src/assets/banorte.png",
    keys: ["BANORTE CLABE TRADICIONAL"],
    color: "#22d3ee",
  },
  bbva: {
    src: "/src/assets/bbva.png",
    keys: ["BBVA CLABE INTERBANCARIA"],
    color: "#3b82f6",
  },
  santander: {
    src: "/src/assets/santander.png",
    keys: ["SANTANDER CLABE TRADICIONAL", "SANTANDER TRADICIONAL REINTENTO "],
    color: "#d946ef",
  },
};

export default function ClusterPanel({ mini = false }: ClusterPanelProps) {
  const [clusters, setClusters] = useState<ClusterPoint[]>([]);
  const [clusterCenters, setClusterCenters] = useState<Record<string, ClusterCenter>>({});
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRefs = useRef<Record<string, HTMLImageElement | null>>({});

  const [tooltipData, setTooltipData] = useState<{
    x: number;
    y: number;
    dy: number;
    clusterName: string;
    logoName: string;
    propor: number;
  } | null>(null);

  useEffect(() => {
    const parsed: ClusterPoint[] = [];
    const allX: number[] = [], allY: number[] = [];
    const PADDING = 0.15;

    Object.values(clusterData).forEach((cluster: any) => {
      allX.push(...cluster.x);
      allY.push(...cluster.y);
    });

    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);

    const scale = (val: number, min: number, max: number) =>
      ((val - min) / (max - min)) * (1 - 2 * PADDING) + PADDING;

    const centers: Record<string, ClusterCenter> = {};

    Object.entries(clusterData).forEach(([clusterName, data]) => {
      const xValues = data.x.map((x) => scale(x, minX, maxX));
      const yValues = data.y.map((y) => scale(y, minY, maxY));
      const avgX = xValues.reduce((a, b) => a + b, 0) / xValues.length;
      const avgY = yValues.reduce((a, b) => a + b, 0) / yValues.length;
      
      centers[clusterName] = {
        leftPercent: avgX * 100,
        topPercent: (1 - avgY) * 100,
        ponderaciones: data.ponderacion ?? {},
      };

      data.x.forEach((x, i) => {
        const y = data.y[i];
        const normX = scale(x, minX, maxX);
        const normY = scale(y, minY, maxY);

        parsed.push({
          id: `${clusterName}-${i}`,
          cluster: clusterName,
          value: 0,
          color: ["#2563eb", "#059669", "#b91c1c", "#7c3aed"][parseInt(clusterName) % 4],
          type: ["A", "B", "C"][i % 3],
          status: i % 2 === 0 ? "active" : "inactive",
          topPercent: (1 - normY) * 100,
          leftPercent: normX * 100,
          propor: 0,
        });
      });
    });

    setClusters(parsed);
    setClusterCenters(centers);
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-screen overflow-hidden"
      style={{
        backgroundImage:
          "linear-gradient(to right, #e5e7eb 1px, transparent 1px), linear-gradient(to bottom, #e5e7eb 1px, transparent 1px)",
        backgroundSize: "calc(100% / 20) calc(100% / 20)",
      }}
    >
      {/* Logos */}
      <div className="absolute top-0 right-0 h-full flex flex-col justify-around pr-4 space-y-2 z-40">
        {Object.entries(logoMapping).map(([key, { src }]) => (
          <img
            key={key}
            ref={(el) => (imageRefs.current[key] = el)}
            src={src}
            alt={key}
            className="w-20 h-20 object-contain"
          />
        ))}
      </div>

      {/* Líneas punteadas con animación */}
      {Object.entries(clusterCenters).flatMap(([clusterName, center]) => {
        return Object.entries(logoMapping).flatMap(([logoKey, { keys, color }]) => {
          const ponderacionValue = keys.reduce((sum, key) => sum + (center.ponderaciones[key] ?? 0), 0);
          if (!ponderacionValue || !imageRefs.current[logoKey] || !containerRef.current) return [];

          const logoRef = imageRefs.current[logoKey]!;
          const logoRect = logoRef.getBoundingClientRect();
          const containerRect = containerRef.current.getBoundingClientRect();

          const startX = (center.leftPercent / 100) * containerRef.current.offsetWidth;
          const startY = (center.topPercent / 100) * containerRef.current.offsetHeight;
          const endX = logoRect.left + logoRect.width / 2 - containerRect.left;
          const endY = logoRect.top + logoRect.height / 2 - containerRect.top;

          const dx = endX - startX;
          const dy = endY - startY;
          const length = Math.sqrt(dx * dx + dy * dy);
          const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
          const thickness = 2 + Math.min(Math.max(ponderacionValue, 0), 1) * 6;
          const animationDuration = 3 - Math.min(Math.max(ponderacionValue, 0), 1) * 2;

          return (
            <div
              key={`${clusterName}-${logoKey}`}
              className="absolute"
              style={{
                top: `${startY}px`,
                left: `${startX}px`,
                width: `${length}px`,
                height: `${thickness}px`,
                transform: `rotate(${angle}deg)`,
                transformOrigin: "left center",
              }}
              onMouseEnter={() =>
                setTooltipData({
                  x: startX + dx / 2,
                  y: startY + dy / 2,
                  dy,
                  clusterName,
                  logoName: logoKey,
                  propor: ponderacionValue,
                })
              }
              onMouseLeave={() => setTooltipData(null)}
            >
              <div
                style={{
                  width: "100%",
                  height: "100%",
                  backgroundImage: `repeating-linear-gradient(to right, ${color}, ${color} 5px, transparent 5px, transparent 10px)`,
                  backgroundSize: "200% 100%",
                  animation: `dash ${animationDuration}s linear infinite`,
                }}
              />
            </div>
          );
        });
      })}

      {/* Tooltip */}
      {tooltipData && (
       <div
  className="absolute z-50 bg-white text-black text-xs px-3 py-1 rounded shadow border transition-opacity pointer-events-none"
  style={{
    top: `${tooltipData.dy > 0 ? tooltipData.y - 40 : tooltipData.y + 10}px`,
    left: `${tooltipData.x}px`,
    transform: "translateX(-50%)",
    whiteSpace: "nowrap",
  }}
>
  {`Grupo: ${tooltipData.clusterName === "cluster0" ? "No pagarán" : "Sí pagarán"}`}<br />
  {`Banco: ${tooltipData.logoName}`}<br />
  {`Proporción: ${(tooltipData.propor * 100).toFixed(1)}%`}
</div>

      )}

      {/* Puntos cluster con color */}
      {clusters.map((cluster) => (
        <div
          key={cluster.id}
          className="absolute rounded-full"
          style={{
            top: `calc(${cluster.topPercent}% - 3px)`,
            left: `calc(${cluster.leftPercent}% - 3px)`,
            width: "6px",
            height: "6px",
            backgroundColor: cluster.color,
            border: "1px solid #e5e7eb",
          }}
        />
      ))}

      <style>{`
        @keyframes dash {
          0% { background-position: 0 0; }
          100% { background-position: -100% 0; }
        }
      `}</style>
    </div>
  );
}
