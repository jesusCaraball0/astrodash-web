import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  Slider,
  FormControlLabel,
  Checkbox,
  TextField,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Snackbar,
  Alert,
  Chip,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton as MuiIconButton,
  AppBar,
  Toolbar,
  Link as MuiLink,
  Fade
} from '@mui/material';
import Grid from '@mui/material/Grid';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ReferenceLine, Customized } from 'recharts';
import { api, ProcessParams, ProcessResponse, LineListResponse } from '../services/api';
import { ResponsiveContainer, Customized as RechartsCustomized } from 'recharts';
import AnalysisOptionPanel from './AnalysisOptionPanel';
import CloseIcon from '@mui/icons-material/Close';
import { useNavigate, useLocation } from 'react-router-dom';
import { ModelType } from './ModelSelectionDialog';
import { styled } from '@mui/material/styles';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import StarIcon from '@mui/icons-material/Star';
import WhatshotIcon from '@mui/icons-material/Whatshot';
import LensBlurIcon from '@mui/icons-material/LensBlur';
import { keyframes } from '@mui/system';

// Spacey card style
const SpaceCard = styled(Paper)(({ theme }) => ({
  background: 'rgba(30,34,60,0.88)',
  borderRadius: 14,
  boxShadow: '0 2px 8px 0 rgba(80,120,255,0.10)',
  border: '1.5px solid rgba(120,80,200,0.12)',
  backdropFilter: 'blur(2px)',
  padding: theme.spacing(3, 3), // 24px top/bottom, 24px left/right
  marginBottom: theme.spacing(3),
  position: 'relative',
  overflow: 'hidden',
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(2, 1.5),
  },
}));

const SpaceSectionHeader = styled(Typography)(({ theme }) => ({
  fontWeight: 600,
  fontSize: '1.5rem',
  letterSpacing: '0.04em',
  color: '#b3baff',
  textShadow: '0 2px 12px rgba(80,120,255,0.18)',
  marginBottom: theme.spacing(2.5),
  marginTop: theme.spacing(1.5),
  paddingLeft: theme.spacing(0.5),
}));

const SpaceyLabel = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
  fontSize: '1rem',
  color: '#b0b8c9',
  marginBottom: theme.spacing(1.2),
  marginTop: theme.spacing(0.5),
}));

interface SupernovaClassifierProps {
  toggleColorMode: () => void;
  currentMode: 'light' | 'dark';
}

// Template spectrum type with additional metadata
type TemplateSpectrum = {
  wave: number[];
  flux: number[];
  sn_type: string;
  age_bin: string;
  color?: string;
  visible?: boolean;
};

// Template overlay configuration
interface TemplateOverlay {
  sn_type: string;
  age_bin: string;
  visible: boolean;
  color: string;
  spectrum: TemplateSpectrum | null;
}

// Template colors for different overlays (move outside component to avoid useEffect dependency warning)
const templateColors = [
  '#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
];

// Canonical order of element/ion names, matching sneLineList.txt
const elementOrder = [
  'H_Balmer', 'H_Paschen', 'HeI', 'HeII', 'CII', 'CIII', 'CIV', 'NII', 'NIII', 'NIV', 'NV',
  'OI', '[OI]', 'OII', '[OII]', '[OIII]', 'OV', 'OVI', 'NaI', 'MgI', 'MgII',
  'SiII', 'SII', 'CaII', 'CaII [H&K]', 'CaII [IR-trip]', '[CaII]', 'FeII', 'FeIII', 'T1', 'T2'
];

// Add this type above the component definition (or near imports):
type ChartContextProps = {
  xAxisMap: any;
  yAxisMap: any;
  width: number;
  height: number;
};

const SupernovaClassifier: React.FC<SupernovaClassifierProps> = ({ toggleColorMode, currentMode }) => {
  const navigate = useNavigate();
  const location = useLocation();

  // Get the selected model from navigation state, default to 'dash'
  let selectedModel: ModelType = location.state?.model;
  if (!selectedModel) {
    const stored = localStorage.getItem('astrodash_selected_model');
    if (stored) {
      try {
        selectedModel = JSON.parse(stored);
      } catch {
        selectedModel = 'dash';
      }
    } else {
      selectedModel = 'dash';
    }
  }

  // State for file selection
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('Select SN File...');

  // State for priors
  const [knownZ, setKnownZ] = useState<boolean>(false);
  const [zValue, setZValue] = useState<string>('');
  const [minWave, setMinWave] = useState<string>('3500');
  const [maxWave, setMaxWave] = useState<string>('10000');
  const [smoothing, setSmoothing] = useState<number>(0);
  const [calculateRlap, setCalculateRlap] = useState<boolean>(false);

  // State for analysis
  const [snType, setSnType] = useState<string>('');
  const [age, setAge] = useState<string>('');
  const [spectrumData, setSpectrumData] = useState<ProcessResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Best matches state
  const [bestMatches, setBestMatches] = useState<Array<{
    type: string;
    age: string;
    probability: number;
    redshift?: number;
    rlap?: number;
    reliable: boolean;
  }>>([]);

  // Add OSC reference state
  const [oscRef, setOscRef] = useState<string>('');

  // Add state for dynamic dropdown options
  const [snTypeOptions, setSnTypeOptions] = useState<string[]>([]);
  const [ageOptions, setAgeOptions] = useState<string[]>([]);

  // Template overlay state
  const [templateOverlays, setTemplateOverlays] = useState<TemplateOverlay[]>([]);
  const [showTemplates, setShowTemplates] = useState<boolean>(false);

  // Add state for error handling
  const [error, setError] = useState<string | null>(null);
  const [errorOpen, setErrorOpen] = useState(false);

  // State for line list (element/ion lines)
  const [lineList, setLineList] = useState<LineListResponse>({});
  const [visibleLines, setVisibleLines] = useState<string[]>([]);
  const [showElementLines, setShowElementLines] = useState<boolean>(false);
  const lineColors = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#dede00', '#00ced1',
    '#8b0000', '#4682b4', '#228b22', '#800080', '#ffa500',
    '#b8860b', '#ff69b4', '#708090', '#bdb76b', '#20b2aa',
    '#e9967a', '#00fa9a', '#8a2be2', '#ff6347', '#4682b4',
    '#ffd700', '#adff2f', '#dc143c', '#00bfff', '#9932cc', '#ff4500'
  ]; // Extended to 31 for all elements

  // Add state for customizing plot
  const [customizeOpen, setCustomizeOpen] = useState(false);

  // Add pulse animation for the top Chip
  const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 #ffe066cc; }
  40% { box-shadow: 0 0 24px 12px #ffe066cc; }
  80% { box-shadow: 0 0 12px 6px #ffe06688; }
  100% { box-shadow: 0 0 0 0 #ffe06600; }
`;

  // Helper function to interpolate template data to match spectrum wavelengths
  const interpolateTemplate = (template: TemplateSpectrum, spectrumWavelengths: number[]): (number | undefined)[] => {
    if (!template || !template.wave || !template.flux || template.wave.length === 0) {
      return new Array(spectrumWavelengths.length).fill(undefined);
    }

    const interpolated = spectrumWavelengths.map(wave => {
      // Find the two closest template wavelengths
      let lowerIdx = -1;
      let upperIdx = -1;

      for (let i = 0; i < template.wave.length; i++) {
        if (template.wave[i] <= wave) {
          lowerIdx = i;
        } else {
          upperIdx = i;
          break;
        }
      }

      // If wave is outside template range, return undefined
      if (lowerIdx === -1 || upperIdx === -1) {
        return undefined;
      }

      // Linear interpolation
      const lowerWave = template.wave[lowerIdx];
      const upperWave = template.wave[upperIdx];
      const lowerFlux = template.flux[lowerIdx];
      const upperFlux = template.flux[upperIdx];

      const ratio = (wave - lowerWave) / (upperWave - lowerWave);
      return lowerFlux + ratio * (upperFlux - lowerFlux);
    });

    return interpolated;
  };

  // Add useEffect to log spectrumData changes
  useEffect(() => {
    if (spectrumData) {
      console.log('spectrumData updated:', spectrumData);
      if (spectrumData.spectrum && spectrumData.spectrum.y) {
        console.log('First 10 flux values (y):', spectrumData.spectrum.y.slice(0, 10));
        console.log('Min flux:', Math.min(...spectrumData.spectrum.y));
        console.log('Max flux:', Math.max(...spectrumData.spectrum.y));
      }
    }
  }, [spectrumData]);

  // Fetch analysis options on mount
  useEffect(() => {
    api.getAnalysisOptions().then(res => {
      setSnTypeOptions(res.sn_types || []);
      // Handle both old format (flat age_bins) and new format (age_bins_by_type)
      if (res.age_bins_by_type) {
        // New format: extract all unique age bins from all SN types
        const allAgeBins = new Set<string>();
        Object.values(res.age_bins_by_type).forEach((ageBins: any) => {
          if (Array.isArray(ageBins)) {
            ageBins.forEach(bin => allAgeBins.add(bin));
          }
        });
        setAgeOptions(Array.from(allAgeBins).sort());
      } else {
        // Old format: flat list
        setAgeOptions(res.age_bins || []);
      }
    }).catch(error => {
      console.error("Failed to fetch analysis options:", error);
      setError("Failed to load classification data. Please try again later.");
      setErrorOpen(true);
    });
  }, []);

  // When dropdowns change, randomly select a best match and update analysis
  useEffect(() => {
    if (bestMatches.length > 0) {
      // Find all matches that match the selected SN type, age, and host
      const filtered = bestMatches.filter(m =>
        (!snType || m.type === snType) &&
        (!age || m.age === age)
      );
      if (filtered.length > 0) {
        setSnType(filtered[0].type);
        setAge(filtered[0].age);
        // Optionally update plot/analysis here
      }
    }
  }, [snType, age, bestMatches]);

  // Initialize template overlays from best matches
  useEffect(() => {
    if (bestMatches.length > 0 && showTemplates) {
      const newOverlays: TemplateOverlay[] = bestMatches.slice(0, 5).map((match, index) => ({
        sn_type: match.type,
        age_bin: match.age,
        visible: index < 3, // Show first 3 by default
        color: templateColors[index % templateColors.length],
        spectrum: null
      }));
      setTemplateOverlays(newOverlays);
    }
  }, [bestMatches, showTemplates]);

  // Fetch template spectra for overlays
  useEffect(() => {
    if (showTemplates && templateOverlays.length > 0 && spectrumData) {
      templateOverlays.forEach(async (overlay, index) => {
        if (overlay.visible && !overlay.spectrum) {
          try {
            const templateData = await api.getTemplateSpectrum(overlay.sn_type, overlay.age_bin);
            setTemplateOverlays(prev => prev.map((o, i) =>
              i === index ? { ...o, spectrum: templateData } : o
            ));
          } catch (error: any) {
            console.error(`Failed to fetch template for ${overlay.sn_type} ${overlay.age_bin}:`, error);
            setError(
              `Template spectrum unavailable for ${overlay.sn_type} (${overlay.age_bin}). It has been removed from the overlays.`
            );
            setErrorOpen(true);
            // Remove the unavailable overlay
            setTemplateOverlays(prev => prev.filter((_, i) => i !== index));
          }
        }
      });
    }
  }, [templateOverlays, showTemplates, spectrumData]);

  // Fetch line list on mount
  useEffect(() => {
    api.getLineList().then((data) => {
      setLineList(data);
      setVisibleLines(Object.keys(data).slice(0, 5)); // Show first 5 by default
    });
  }, []);

  // Toggle visibility of a line group
  const toggleLineVisibility = (element: string) => {
    setVisibleLines((prev) =>
      prev.includes(element)
        ? prev.filter((el) => el !== element)
        : [...prev, element]
    );
  };

  // Prepare chart data with template overlays
  const getChartData = () => {
    if (!spectrumData || !spectrumData.spectrum) return [];

    const baseData = spectrumData.spectrum.x.map((x: number, i: number) => ({
      x,
      y: spectrumData.spectrum.y[i],
    }));

    if (!showTemplates || templateOverlays.length === 0) {
      return baseData;
    }

    // Add template data to each point
    return baseData.map(point => {
      const templateData: any = { x: point.x, y: point.y };

      templateOverlays.forEach((overlay, index) => {
        if (overlay.visible && overlay.spectrum) {
          const interpolatedFlux = interpolateTemplate(overlay.spectrum, [point.x])[0];
          if (interpolatedFlux !== undefined) {
            templateData[`template_${index}`] = interpolatedFlux;
          }
        }
      });

      return templateData;
    });
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setFileName(event.target.files[0].name);
      setOscRef('');
    }
  };

  const handleOscRefChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setOscRef(event.target.value);
    setSelectedFile(null);
  };

  const handleOscRefKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleProcess();
    }
  };

  const handleProcess = async () => {
    console.log('Starting process with params:', {
      smoothing,
      knownZ,
      zValue: knownZ ? parseFloat(zValue) : undefined,
      minWave: parseInt(minWave),
      maxWave: parseInt(maxWave),
      calculateRlap,
      file: selectedFile ? selectedFile.name : undefined,
      oscRef: oscRef || undefined
    });

    setLoading(true);
    setError(null);
    setErrorOpen(false);
    try {
      let modelType: 'dash' | 'transformer' | undefined = undefined;
      let model_id: string | undefined = undefined;
      if (selectedModel === 'dash' || selectedModel === 'transformer') {
        modelType = selectedModel;
      } else if (typeof selectedModel === 'object' && selectedModel.user) {
        model_id = selectedModel.user;
      }
      const params: any = {
        smoothing,
        knownZ,
        zValue: knownZ ? parseFloat(zValue) : undefined,
        minWave: parseInt(minWave),
        maxWave: parseInt(maxWave),
        calculateRlap,
        file: selectedFile || undefined,
        oscRef: oscRef || undefined,
        ...(modelType ? { modelType } : {}),
        ...(model_id ? { model_id } : {}),
      };

      console.log('Calling API with params:', params);
      const response = await api.processSpectrum(params);
      console.log('Received response:', response);

      // Type guard for error response
      const isErrorResponse = (resp: any): resp is { status?: string; error?: string; message?: string } => {
        return (
          (typeof resp === 'object') &&
          (
            (typeof resp.status === 'string' && resp.status === 'error') ||
            typeof resp.error === 'string' ||
            typeof resp.message === 'string'
          )
        );
      };

      if (isErrorResponse(response)) {
        setError(response.message || response.error || 'An unknown error occurred.');
        setErrorOpen(true);
        return;
      }

      setSpectrumData(response);
      setBestMatches(response.classification.best_matches || []);

      // Clear template overlays when new spectrum is processed
      setTemplateOverlays([]);
      setShowTemplates(false);

    } catch (error) {
      console.error('Error processing spectrum:', error);
      setError('Failed to process spectrum. Please try again.');
      setErrorOpen(true);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setFileName('Select SN File...');
    setOscRef('');
    setSpectrumData(null);
    setBestMatches([]);
    setTemplateOverlays([]);
    setShowTemplates(false);
    setError(null);
  };

  const handleDownload = () => {
    if (!spectrumData) return;

    const dataStr = JSON.stringify(spectrumData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'spectrum_analysis.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleHelp = () => {
    // Implement help functionality
    console.log('Help clicked');
  };

  const toggleTemplateVisibility = (index: number) => {
    setTemplateOverlays(prev => prev.map((overlay, i) =>
      i === index ? { ...overlay, visible: !overlay.visible } : overlay
    ));
  };

  const addTemplateOverlay = () => {
    if (snType && age) {
      const newOverlay: TemplateOverlay = {
        sn_type: snType,
        age_bin: age,
        visible: true,
        color: templateColors[templateOverlays.length % templateColors.length],
        spectrum: null
      };
      setTemplateOverlays(prev => [...prev, newOverlay]);
    }
  };

  const removeTemplateOverlay = (index: number) => {
    setTemplateOverlays(prev => prev.filter((_, i) => i !== index));
  };

  // Compute the min/max for XAxis domain to include only the spectrum data
  const getSpectrumDomain = () => {
    if (spectrumData && spectrumData.spectrum && spectrumData.spectrum.x.length > 0) {
      const min = Math.min(...spectrumData.spectrum.x);
      const max = Math.max(...spectrumData.spectrum.x);
      // Add a small buffer
      const buffer = (max - min) * 0.01;
      return [min - buffer, max + buffer];
    }
    return undefined;
  };

  // Get spectrum x range for filtering lines
  const getSpectrumXRange = () => {
    if (spectrumData && spectrumData.spectrum && spectrumData.spectrum.x.length > 0) {
      const min = Math.min(...spectrumData.spectrum.x);
      const max = Math.max(...spectrumData.spectrum.x);
      return [min, max];
    }
    return [null, null];
  };

  // Helper to get color for an element/ion by canonical order
  const getElementColor = (element: string) => {
    const idx = elementOrder.indexOf(element);
    return lineColors[idx >= 0 ? idx % lineColors.length : 0];
  };

  // Helper to get a short label/abbreviation for an element/ion
  const getElementShortLabel = (element: string) => {
    // Customize as needed for clarity
    if (element.startsWith('H_')) return 'H';
    if (element.startsWith('HeI')) return 'He I';
    if (element.startsWith('HeII')) return 'He II';
    if (element.startsWith('CII')) return 'C II';
    if (element.startsWith('CIII')) return 'C III';
    if (element.startsWith('CIV')) return 'C IV';
    if (element.startsWith('NII')) return 'N II';
    if (element.startsWith('NIII')) return 'N III';
    if (element.startsWith('NIV')) return 'N IV';
    if (element.startsWith('NV')) return 'N V';
    if (element.startsWith('OI')) return 'O I';
    if (element.startsWith('[OI]')) return '[O I]';
    if (element.startsWith('OII')) return 'O II';
    if (element.startsWith('[OII]')) return '[O II]';
    if (element.startsWith('[OIII]')) return '[O III]';
    if (element.startsWith('OV')) return 'O V';
    if (element.startsWith('OVI')) return 'O VI';
    if (element.startsWith('NaI')) return 'Na I';
    if (element.startsWith('MgI')) return 'Mg I';
    if (element.startsWith('MgII')) return 'Mg II';
    if (element.startsWith('SiII')) return 'Si II';
    if (element.startsWith('SII')) return 'S II';
    if (element.startsWith('CaII [H&K]')) return 'Ca II H&K';
    if (element.startsWith('CaII [IR-trip]')) return 'Ca II IR';
    if (element.startsWith('CaII')) return 'Ca II';
    if (element.startsWith('[CaII]')) return '[Ca II]';
    if (element.startsWith('FeII')) return 'Fe II';
    if (element.startsWith('FeIII')) return 'Fe III';
    if (element.startsWith('T1')) return 'T1';
    if (element.startsWith('T2')) return 'T2';
    // Fallback: first capital letter(s)
    return element.replace(/[^A-Z]/g, '').slice(0, 3);
  };

  // Custom legend for element/ion lines
  const ElementLineLegend = ({ visibleLines, lineList }: { visibleLines: string[], lineList: LineListResponse }) => (
    <div style={{ position: 'absolute', top: 10, left: 40, display: 'flex', flexDirection: 'row', gap: 16, pointerEvents: 'none', zIndex: 2 }}>
      {visibleLines.map((element) => (
        <span key={element} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 18, height: 6, background: getElementColor(element), display: 'inline-block', borderRadius: 2 }} />
          <span style={{ fontSize: 12, color: '#333', background: 'rgba(255,255,255,0.8)', padding: '0 4px', borderRadius: 2 }}>{element}</span>
        </span>
      ))}
    </div>
  );

  // Legend for visible element/ion lines and template overlays
  const GraphLegend = () => {
    let items: { label: string; color: string; type: 'element' | 'template' }[] = [];
    if (showElementLines) {
      items = items.concat(
        visibleLines.map((element) => ({
          label: element,
          color: getElementColor(element),
          type: 'element' as const,
        }))
      );
    }
    if (showTemplates) {
      items = items.concat(
        templateOverlays
          .filter((overlay) => overlay.visible && overlay.spectrum)
          .map((overlay) => ({
            label: `${overlay.sn_type} ${overlay.age_bin}`,
            color: overlay.color,
            type: 'template' as const,
          }))
      );
    }
    if (items.length === 0) return null;
    return (
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 2, alignItems: 'center', justifyContent: 'center' }}>
        {items.map((item) => (
          <Box key={item.label + item.color} sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1, py: 0.5, borderRadius: 1, background: '#f5f5f5', minWidth: 0 }}>
            <Box sx={{ width: 18, height: 6, background: item.color, borderRadius: 2, mr: 1 }} />
            <Typography variant="body2" sx={{ fontWeight: 500, color: '#333', whiteSpace: 'nowrap' }}>{item.label}</Typography>
            {item.type === 'template' && (
              <Typography variant="caption" sx={{ color: '#888', ml: 0.5 }}>(template)</Typography>
            )}
            {item.type === 'element' && (
              <Typography variant="caption" sx={{ color: '#888', ml: 0.5 }}>(line)</Typography>
            )}
          </Box>
        ))}
      </Box>
    );
  };

  // Add a helper for SN type chip color and icon
  const snTypeChipProps = (type: string, isTop: boolean) => {
    switch (type.toLowerCase()) {
      case 'ia':
        return { color: '#ffd700', icon: <StarIcon sx={{ fontSize: 20 }} />, label: 'Ia' };
      case 'ib/c':
        return { color: '#ff7043', icon: <WhatshotIcon sx={{ fontSize: 20 }} />, label: 'Ib/c' };
      case 'ii':
        return { color: '#64b5f6', icon: <LensBlurIcon sx={{ fontSize: 20 }} />, label: 'II' };
      default:
        return { color: '#b3baff', icon: <StarIcon sx={{ fontSize: 20 }} />, label: type };
    }
  };

  // In the Best Matches section, add pulse animation to the top Chip when bestMatches changes
  const [pulseKey, setPulseKey] = React.useState(0);
  React.useEffect(() => {
    if (bestMatches.length > 0) {
      setPulseKey((k) => k + 1);
    }
  }, [bestMatches]);

  return (
    <Container maxWidth="xl" sx={{ py: 2, mt: 2 }}>
      {/* Header */}
      <AppBar position="static" elevation={0} sx={{
        background: 'rgba(24, 28, 48, 0.55)',
        boxShadow: '0 8px 32px 0 rgba(80,120,255,0.18)',
        border: 'none',
        backdropFilter: 'blur(12px) saturate(1.2)',
        zIndex: 1300,
        left: 0,
        right: 0,
        top: 0,
        px: 0,
        mb: 4,
      }}>
        <Toolbar sx={{ minHeight: 56, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', px: 0 }}>
          {/* Floating back button */}
          <IconButton onClick={() => navigate('/')} color="inherit" sx={{
            position: 'absolute',
            left: 24,
            top: '50%',
            transform: 'translateY(-50%)',
            background: 'rgba(40, 44, 80, 0.7)',
            boxShadow: '0 2px 8px 0 rgba(80,120,255,0.18)',
            borderRadius: '50%',
            p: 1.2,
            '&:hover': { background: 'rgba(80,120,255,0.18)' },
          }}>
            <ArrowBackIcon sx={{ fontSize: 28, color: '#b3baff' }} />
          </IconButton>
          {/* Glowing Astrodash icon */}
          <Box sx={{
            width: 38,
            height: 38,
            borderRadius: '50%',
            background: 'radial-gradient(circle at 60% 40%, #a18cd1 0%, #fbc2eb 100%)',
            boxShadow: '0 0 16px 4px #a18cd1, 0 0 32px 8px #fbc2eb44',
            display: 'inline-block',
            mr: 2,
          }} />
          {/* Centered cosmic-gradient title */}
          <Typography
            variant="h5"
            sx={{
              fontWeight: 800,
              letterSpacing: '0.08em',
              background: 'linear-gradient(90deg, #b3baff 0%, #fbc2eb 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 2px 16px #2a2a4a',
              mx: 2,
              textAlign: 'center',
              flex: 1,
            }}
          >
            Astrodash Supernova Classifier
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Error Snackbar */}
      <Snackbar open={errorOpen} autoHideDuration={8000} onClose={() => setErrorOpen(false)} anchorOrigin={{ vertical: 'top', horizontal: 'center' }}>
        <Alert onClose={() => setErrorOpen(false)} severity="error" sx={{ width: '100%' }} variant="filled">
          {error}
        </Alert>
      </Snackbar>

      {/* Update the main Grid layout for responsiveness */}
      <Grid container spacing={2}>
        {/* Left Panel */}
        <Grid item xs={12} md={3}>
          <SpaceCard>
            <SpaceSectionHeader>Select Spectrum</SpaceSectionHeader>

            {/* Add OSC Reference Input */}
            <Box sx={{ mb: 2 }}>
              <TextField
                fullWidth
                label="OSC Reference"
                value={oscRef}
                onChange={handleOscRefChange}
                onKeyPress={handleOscRefKeyPress}
                placeholder="e.g., osc-sn2002er-8"
                size="small"
                sx={{ mb: 1 }}
              />
              <Typography variant="caption" color="text.secondary">
                Enter an OSC reference (e.g., osc-sn2002er-8) or upload a file
              </Typography>
            </Box>

            {/* Existing file upload UI */}
            <Box sx={{ mb: 2 }}>
              <input
                accept=".fits,.dat,.txt"
                style={{ display: 'none' }}
                id="file-input"
                type="file"
                onChange={handleFileSelect}
              />
              <Box sx={{ display: 'flex', gap: 1 }}>
                <label htmlFor="file-input">
                  <Button
                    variant="contained"
                    component="span"
                    fullWidth
                    sx={{
                      transition: 'transform 0.15s, box-shadow 0.15s',
                      boxShadow: '0 2px 8px 0 #90caf9aa',
                      '&:hover': {
                        transform: 'scale(1.04)',
                        boxShadow: '0 4px 16px 0 #90caf9cc',
                      },
                      '&:active': {
                        transform: 'scale(0.98)',
                        boxShadow: '0 1px 4px 0 #90caf988',
                      },
                    }}
                  >
                    Browse
                  </Button>
                </label>
                <Button
                  variant="outlined"
                  onClick={handleClear}
                  disabled={!selectedFile && !oscRef}
                  sx={{
                    transition: 'transform 0.15s, box-shadow 0.15s',
                    boxShadow: '0 2px 8px 0 #b3baff55',
                    '&:hover': {
                      transform: 'scale(1.04)',
                      boxShadow: '0 4px 16px 0 #b3baff99',
                    },
                    '&:active': {
                      transform: 'scale(0.98)',
                      boxShadow: '0 1px 4px 0 #b3baff66',
                    },
                  }}
                >
                  Clear
                </Button>
              </Box>
              <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
                {fileName}
              </Typography>
            </Box>
          </SpaceCard>

          <SpaceCard>
            <SpaceSectionHeader>Priors</SpaceSectionHeader>
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={knownZ}
                    onChange={(e) => setKnownZ(e.target.checked)}
                  />
                }
                label="Known Redshift"
              />
              {knownZ && (
                <TextField
                  size="small"
                  value={zValue}
                  onChange={(e) => setZValue(e.target.value)}
                  sx={{ ml: 2 }}
                />
              )}
            </Box>

            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Wavelength Range</Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  size="small"
                  label="Min wave"
                  value={minWave}
                  onChange={(e) => setMinWave(e.target.value)}
                  fullWidth
                />
                <TextField
                  size="small"
                  label="Max wave"
                  value={maxWave}
                  onChange={(e) => setMaxWave(e.target.value)}
                  fullWidth
                />
              </Box>
            </Box>

            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Smooth</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TextField
                  size="small"
                  value={smoothing}
                  onChange={(e) => setSmoothing(Number(e.target.value))}
                  sx={{ width: 60, mr: 2 }}
                />
                <Slider
                  value={typeof smoothing === 'number' ? smoothing : 0}
                  onChange={(_event, newValue) => setSmoothing(newValue as number)}
                  aria-labelledby="smooth-slider"
                  valueLabelDisplay="auto"
                  step={1}
                  marks
                  min={0}
                  max={50}
                />
              </Box>
            </Box>

            <FormControlLabel
              control={
                <Checkbox
                  checked={calculateRlap}
                  onChange={(e) => setCalculateRlap(e.target.checked)}
                />
              }
              label="Calculate rlap scores"
            />

            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleProcess}
                disabled={loading || (!selectedFile && !oscRef)}
                fullWidth
                sx={{
                  py: 1.5,
                  transition: 'transform 0.15s, box-shadow 0.15s',
                  boxShadow: '0 2px 8px 0 #ffd70033',
                  '&:hover': {
                    transform: 'scale(1.04)',
                    boxShadow: '0 4px 16px 0 #ffd70066',
                  },
                  '&:active': {
                    transform: 'scale(0.98)',
                    boxShadow: '0 1px 4px 0 #ffd70044',
                  },
                }}
              >
                {loading ? 'Processing...' : 'Process Spectrum'}
              </Button>
            </Box>
          </SpaceCard>
        </Grid>

        {/* Right Panel */}
        <Grid item xs={12} md={9}>
          <Grid container spacing={2}>
            {/* Best Matches Section */}
            <Grid item xs={12}>
              <SpaceCard>
                <SpaceSectionHeader>Best Matches</SpaceSectionHeader>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Box sx={{ width: '100%' }}>
                    <Fade in={bestMatches.length > 0} timeout={600}>
                      <List>
                        {bestMatches.map((match, index) => {
                          const isTop = index === 0;
                          const { color: chipColor, label: chipLabel } = snTypeChipProps(match.type, isTop);
                          return (
                            <ListItem key={index} sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                              <Chip
                                key={`top-chip-${pulseKey}`}
                                label={chipLabel + (match.age ? ` (${match.age})` : '')}
                                sx={{
                                  background: isTop ? `linear-gradient(90deg, ${chipColor} 60%, #fffde4 100%)` : chipColor,
                                  color: isTop ? '#222' : '#fff',
                                  fontWeight: isTop ? 700 : 500,
                                  fontSize: { xs: '0.95rem', sm: isTop ? '1.1rem' : '1rem' },
                                  px: { xs: 1.2, sm: isTop ? 2 : 1.2 },
                                  py: { xs: 0.7, sm: isTop ? 1.2 : 0.7 },
                                  minWidth: { xs: 60, sm: 80 },
                                  boxShadow: isTop ? '0 0 16px 4px #ffe066cc' : 'none',
                                  border: isTop ? '2px solid #ffd700' : 'none',
                                  animation: isTop ? `${pulse} 1.8s` : undefined,
                                  animationIterationCount: isTop ? 1 : undefined,
                                }}
                              />
                              <Typography variant="body2" sx={{ color: '#b0b8c9', ml: 1 }}>
                                Prob: {(match.probability * 100).toFixed(1)}%
                                {match.rlap !== undefined && match.rlap !== null &&
                                  <> | RLAP: {match.rlap}</>
                                }
                              </Typography>
                            </ListItem>
                          );
                        })}
                      </List>
                    </Fade>
                  </Box>
                </Box>
              </SpaceCard>
            </Grid>

            {/* Analysis Section */}
            <Grid item xs={12}>
              <SpaceCard>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Analyse selection</Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      onClick={() => setCustomizeOpen(true)}
                      disabled={!spectrumData}
                    >
                      Customize Plot
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={handleDownload}
                      disabled={!spectrumData}
                    >
                      Save
                    </Button>
                  </Box>
                </Box>

                {/* Spectrum Plot Section */}
                <Box>
                  <Fade in={!!spectrumData} timeout={700}>
                    <div className="mt-8">
                      <h2 className="text-xl font-bold mb-4">Spectrum Plot</h2>
                      <div className="bg-white p-4 rounded-lg shadow">
                        <ResponsiveContainer width="100%" height={400}>
                          <LineChart
                            key={spectrumData ? `spectrum-${spectrumData.spectrum.x[0]}` : 'no-spectrum'}
                            data={getChartData()}
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="x"
                              type="number"
                              label={{ value: 'Wavelength (Å)', position: 'insideBottom', offset: -5 }}
                              domain={getSpectrumDomain()}
                            />
                            <YAxis
                              label={{ value: 'Flux', angle: -90, position: 'insideLeft' }}
                            />
                            <RechartsTooltip
                              formatter={(value: number, name: string, props: any) => {
                                if (name.startsWith('Template')) {
                                  return [`${value?.toFixed(6) || 'N/A'}`, name];
                                }
                                return [`${props.payload.y?.toFixed(6) || 'N/A'}`, 'Observed'];
                              }}
                              labelFormatter={(label: number) => `Wavelength: ${label.toFixed(2)} Å`}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} name="Observed" />
                            {templateOverlays.map((overlay, index) => (
                              overlay.visible && overlay.spectrum && (
                                <Line
                                  key={`template-${index}`}
                                  type="monotone"
                                  dataKey={`template_${index}`}
                                  stroke={overlay.color}
                                  dot={false}
                                  name={`${overlay.sn_type} ${overlay.age_bin}`}
                                />
                              )
                            ))}
                            {showElementLines && Object.keys(lineList).length > 0 && visibleLines.length > 0 && (() => {
                              const [minX, maxX] = getSpectrumXRange();
                              return visibleLines.flatMap((element) =>
                                (lineList[element] || [])
                                  .filter((line) => minX !== null && maxX !== null && line >= minX && line <= maxX)
                                  .map((line, i) => (
                                    <ReferenceLine
                                      key={`line-${element}-${line}`}
                                      x={line}
                                      stroke={getElementColor(element)}
                                      strokeDasharray="3 3"
                                    />
                                  ))
                              );
                            })()}
                            {showElementLines && Object.keys(lineList).length > 0 && visibleLines.length > 0 && (
                              <Customized
                                component={(props: any) => {
                                  const { xAxisMap, yAxisMap } = props;
                                  const xAxis = xAxisMap[Object.keys(xAxisMap)[0]];
                                  const yAxis = yAxisMap[Object.keys(yAxisMap)[0]];
                                  if (!xAxis || !yAxis) return <g />;
                                  const yTop = yAxis.y + 8;
                                  const [minX, maxX] = getSpectrumXRange();
                                  return (
                                    <g>
                                      {visibleLines.map((element) => {
                                        const lines = (lineList[element] || []).filter((line) => minX !== null && maxX !== null && line >= minX && line <= maxX);
                                        if (lines.length === 0) return null;
                                        const xPx = xAxis.scale(lines[0]);
                                        return (
                                          <text
                                            key={`label-${element}-${lines[0]}`}
                                            x={xPx + 4}
                                            y={yTop}
                                            fontSize={11}
                                            fill={getElementColor(element)}
                                            fontWeight="bold"
                                            textAnchor="start"
                                            style={{ pointerEvents: 'none', userSelect: 'none' }}
                                          >
                                            {getElementShortLabel(element)}
                                          </text>
                                        );
                                      })}
                                    </g>
                                  );
                                }}
                              />
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                        {/* Dynamic legend for lines and templates */}
                        <GraphLegend />
                      </div>
                    </div>
                  </Fade>
                </Box>
              </SpaceCard>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Add the Dialog for customizing plot options */}
      <Dialog open={customizeOpen} onClose={() => setCustomizeOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ m: 0, p: 2 }}>
          Customize Plot
          <MuiIconButton
            aria-label="close"
            onClick={() => setCustomizeOpen(false)}
            sx={{ position: 'absolute', right: 8, top: 8, color: (theme) => theme.palette.grey[500] }}
          >
            <CloseIcon />
          </MuiIconButton>
        </DialogTitle>
        <DialogContent dividers>
          {spectrumData && bestMatches.length > 0 && (
            <AnalysisOptionPanel
              title="Template Overlays"
              controlRow={
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={showTemplates}
                      onChange={(e) => setShowTemplates(e.target.checked)}
                    />
                  }
                  label="Show Templates"
                />
              }
            >
              {showTemplates && (
                <>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Overlay template spectra from best matches or add custom templates:
                    </Typography>
                    {/* Best Match Templates */}
                    {templateOverlays.length > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Best Match Templates:
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                          {templateOverlays.map((overlay, index) => (
                            <Chip
                              key={index}
                              label={`${overlay.sn_type} ${overlay.age_bin}`}
                              onDelete={() => removeTemplateOverlay(index)}
                              onClick={() => toggleTemplateVisibility(index)}
                              icon={overlay.visible ? <VisibilityIcon /> : <VisibilityOffIcon />}
                              sx={{
                                backgroundColor: overlay.visible ? overlay.color : '#f5f5f5',
                                color: overlay.visible ? 'white' : 'inherit',
                                '&:hover': {
                                  backgroundColor: overlay.visible ? overlay.color : '#e0e0e0',
                                }
                              }}
                            />
                          ))}
                        </Stack>
                      </Box>
                    )}
                    {/* Add Custom Template */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>SN Type</InputLabel>
                        <Select value={snType} onChange={(e) => setSnType(e.target.value)}>
                          {snTypeOptions.map((type) => (
                            <MenuItem key={type} value={type}>
                              {type}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>Age</InputLabel>
                        <Select value={age} onChange={(e) => setAge(e.target.value)}>
                          {ageOptions.map((ageBin) => (
                            <MenuItem key={ageBin} value={ageBin}>
                              {ageBin}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={addTemplateOverlay}
                        disabled={!snType || !age}
                      >
                        Add
                      </Button>
                    </Box>
                  </Box>
                  {/* Template Legend */}
                  {templateOverlays.filter(o => o.visible).length > 0 && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Visible Templates:
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {templateOverlays
                          .filter(overlay => overlay.visible)
                          .map((overlay, index) => (
                            <Box
                              key={index}
                              sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 0.5,
                                p: 0.5,
                                border: `2px solid ${overlay.color}`,
                                borderRadius: 1,
                                backgroundColor: `${overlay.color}20`
                              }}
                            >
                              <Box
                                sx={{
                                  width: 12,
                                  height: 2,
                                  backgroundColor: overlay.color
                                }}
                              />
                              <Typography variant="caption">
                                {overlay.sn_type} {overlay.age_bin}
                              </Typography>
                            </Box>
                          ))}
                      </Box>
                    </Box>
                  )}
                </>
              )}
            </AnalysisOptionPanel>
          )}
          {spectrumData && Object.keys(lineList).length > 0 && (
            <AnalysisOptionPanel
              title="Element/Ion Lines"
              controlRow={
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={showElementLines}
                      onChange={(e) => setShowElementLines(e.target.checked)}
                    />
                  }
                  label="Show Element/Ion Lines"
                  sx={{ ml: 2 }}
                />
              }
            >
              {showElementLines && (
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {Object.keys(lineList).map((element) => (
                    <Chip
                      key={element}
                      label={element}
                      onClick={() => toggleLineVisibility(element)}
                      sx={{
                        backgroundColor: visibleLines.includes(element) ? getElementColor(element) : '#f5f5f5',
                        color: visibleLines.includes(element) ? 'white' : 'inherit',
                        border: visibleLines.includes(element) ? `2px solid ${getElementColor(element)}` : '1px solid #ccc',
                        fontWeight: visibleLines.includes(element) ? 700 : 400,
                        cursor: 'pointer',
                      }}
                      variant={visibleLines.includes(element) ? 'filled' : 'outlined'}
                    />
                  ))}
                </Stack>
              )}
            </AnalysisOptionPanel>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCustomizeOpen(false)} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Footer */}
      <Box component="footer" sx={{
        width: '100%',
        background: 'rgba(20, 24, 40, 0.85)',
        color: '#b3baff',
        py: 1.2,
        px: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '1rem',
        borderTop: '1.5px solid rgba(120,80,200,0.12)',
      }}>
        <MuiLink href="/docs" color="inherit" underline="hover" sx={{ mx: 1 }}>
          Docs
        </MuiLink>
        |
        <MuiLink href="https://github.com/astrodash/astrodash-api" color="inherit" underline="hover" sx={{ mx: 1 }}>
          GitHub
        </MuiLink>
        |
        <span style={{ opacity: 0.7, marginLeft: 8 }}>© {new Date().getFullYear()} Astrodash</span>
      </Box>
    </Container>
  );
};

export default SupernovaClassifier;
