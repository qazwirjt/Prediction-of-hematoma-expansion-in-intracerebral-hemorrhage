import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import tempfile
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import yaml
import json
import re

# conda activate radiomics_app
# streamlit run 8.py


# --- 0. 应用配置和全局变量 ---
st.set_page_config(layout="wide", page_title="影像组学模型预测与SHAP分析 - 增强版")

# --- 关键路径和文件名 ---
SERIALIZED_MODELS_DIR = "serialized_models"
STANDARDIZATION_PARAMS_PATH = "standardization_parameters.csv"
SHAP_BACKGROUND_DATA_PATH = "shap_background_data.csv"
RADIOMICS_PARAMS_PATH = "params_wavelet.yaml"

# --- 初始化session state ---
if 'radiomic_features' not in st.session_state:
    st.session_state.radiomic_features = {}
if 'clinical_features' not in st.session_state:
    st.session_state.clinical_features = {}
if 'all_extracted_features' not in st.session_state:
    st.session_state.all_extracted_features = {}
if 'show_standardization_details' not in st.session_state:
    st.session_state.show_standardization_details = False

# --- 1. 辅助函数和加载函数 ---
@st.cache_data
def load_csv_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"CSV文件 '{path}' 未找到。")
        return None
    except Exception as e:
        st.error(f"加载CSV文件 '{path}' 时出错: {e}")
        return None

@st.cache_data
def load_standardization_parameters(path):
    """加载标准化参数CSV文件"""
    try:
        params_df = pd.read_csv(path)
        
        standardization_info = {
            'params_df': params_df,
            'continuous_features': params_df[params_df['feature_type'] == 'continuous']['feature_name'].tolist(),
            'binary_features': params_df[params_df['feature_type'] == 'binary']['feature_name'].tolist(),
            'mean_dict': dict(zip(
                params_df[params_df['feature_type'] == 'continuous']['feature_name'],
                params_df[params_df['feature_type'] == 'continuous']['mean']
            )),
            'std_dict': dict(zip(
                params_df[params_df['feature_type'] == 'continuous']['feature_name'],
                params_df[params_df['feature_type'] == 'continuous']['std']
            ))
        }
        
        return standardization_info
    except FileNotFoundError:
        st.error(f"标准化参数文件 '{path}' 未找到。")
        return None
    except Exception as e:
        st.error(f"加载标准化参数文件时出错: {e}")
        return None

@st.cache_data
def load_radiomics_params(path):
    """加载PyRadiomics参数配置文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        st.error(f"PyRadiomics配置文件 '{path}' 未找到。")
        return None
    except Exception as e:
        st.error(f"加载PyRadiomics配置文件时出错: {e}")
        return None

@st.cache_resource
def load_model_file(path):
    """加载模型文件"""
    try:
        import joblib
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"模型文件 '{path}' 未找到。")
        return None
    except Exception as e:
        st.error(f"加载模型文件 '{path}' 时出错: {e}")
        return None

# 加载必要的资源
standardization_info = load_standardization_parameters(STANDARDIZATION_PARAMS_PATH)
shap_background_df = load_csv_data(SHAP_BACKGROUND_DATA_PATH)
radiomics_params = load_radiomics_params(RADIOMICS_PARAMS_PATH)

# 检查标准化参数是否成功加载
if standardization_info is not None:
    st.sidebar.success(f"✓ 成功加载标准化参数：{len(standardization_info['continuous_features'])} 个连续特征，{len(standardization_info['binary_features'])} 个二元特征")
else:
    st.sidebar.error("⚠️ 无法加载标准化参数文件")

# --- 特征分类函数 ---
def classify_features(feature_names):
    """基于标准化参数文件中的信息分类特征"""
    if standardization_info is None:
        radiomic_keywords = [
            'original_', 'wavelet_', 'wavelet-', 'log_', 'log-sigma-', 
            'logarithm_', 'gradient_', 'gradient-', 'exponential_', 
            'square_', 'squareroot_', 'shape_', 'shape2D_',
            'firstorder_', 'glcm_', 'glrlm_', 'glszm_', 
            'ngtdm_', 'gldm_', 'glrm_'
        ]
        
        radiomic_features = []
        clinical_features = []
        
        for feature in feature_names:
            if feature.upper() == 'STATUS':
                continue
            
            is_radiomic = any(keyword in feature.lower() for keyword in radiomic_keywords)
            
            if not is_radiomic:
                pyradiomics_patterns = [
                    r'^(log|wavelet|gradient|exponential|square|logarithm|original)-.*_(firstorder|glcm|glrlm|glszm|ngtdm|gldm|shape)',
                    r'^(log-sigma-[\d-]+mm-\dD)_',
                    r'^wavelet-[LH]+_'
                ]
                for pattern in pyradiomics_patterns:
                    if re.match(pattern, feature, re.IGNORECASE):
                        is_radiomic = True
                        break
            
            if is_radiomic:
                radiomic_features.append(feature)
            else:
                clinical_features.append(feature)
    else:
        all_features_in_params = standardization_info['continuous_features'] + standardization_info['binary_features']
        radiomic_features = []
        clinical_features = []
        
        for feature in feature_names:
            if feature.upper() == 'STATUS':
                continue
            
            if feature in all_features_in_params:
                radiomic_keywords = ['original_', 'wavelet', 'log-sigma']
                if any(keyword in feature.lower() for keyword in radiomic_keywords):
                    radiomic_features.append(feature)
                else:
                    clinical_features.append(feature)
            else:
                clinical_features.append(feature)
    
    return radiomic_features, clinical_features

def identify_variable_type(series):
    """识别变量类型：二分类或连续型"""
    non_null_values = series.dropna()
    
    if len(non_null_values) == 0:
        return 'continuous', None
    
    unique_values = non_null_values.unique()
    
    if len(unique_values) == 2:
        return 'binary', sorted(unique_values.tolist())
    
    if len(unique_values) <= 5 and all(isinstance(x, (int, np.integer)) for x in unique_values):
        if set(unique_values).issubset({0, 1}):
            return 'binary', sorted(unique_values.tolist())
    
    return 'continuous', None

# 获取特征分类
if shap_background_df is not None:
    FEATURE_ORDER = list(shap_background_df.columns)
    FEATURE_ORDER = [f for f in FEATURE_ORDER if f.upper() != 'STATUS']
    RADIOMIC_FEATURES, CLINICAL_FEATURES = classify_features(FEATURE_ORDER)
    
    with st.sidebar:
        st.info(f"检测到的特征统计:\n- 影像组学特征: {len(RADIOMIC_FEATURES)} 个\n- 临床特征: {len(CLINICAL_FEATURES)} 个")
else:
    FEATURE_ORDER = []
    RADIOMIC_FEATURES = []
    CLINICAL_FEATURES = []

# --- 2. 影像特征提取函数 ---
def extract_radiomics_features(image_path, mask_path, params):
    """
    使用PyRadiomics从NIfTI图像中提取影像组学特征
    """
    try:
        # 根据是否有参数文件来初始化特征提取器
        if params:
            # 保存参数到临时YAML文件（如果params是字典）
            if isinstance(params, dict):
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_yaml:
                    yaml.dump(params, temp_yaml)
                    temp_yaml_path = temp_yaml.name
                
                # 使用临时YAML文件初始化
                extractor = featureextractor.RadiomicsFeatureExtractor(temp_yaml_path)
                
                # 删除临时文件
                os.unlink(temp_yaml_path)
            else:
                # 如果params是文件路径，直接使用
                extractor = featureextractor.RadiomicsFeatureExtractor(params)
        else:
            # 没有参数文件时使用默认初始化
            extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # 关键步骤：强制启用所有特征类别（与训练代码保持一致）
        extractor.enableAllFeatures()
        st.info("已启用所有特征类别")
        
        # 记录启用的图像类型
        enabled_image_types = extractor.enabledImagetypes
        if enabled_image_types:
            st.info(f"启用的图像类型: {list(enabled_image_types.keys())}")
        
        # 提取特征
        st.info("正在提取影像组学特征，请稍候...")
        feature_vector = extractor.execute(image_path, mask_path)
        
        # 转换为字典，只保留特征值（排除诊断信息）
        feature_dict = {}
        for key, value in feature_vector.items():
            if not key.startswith('diagnostics_'):
                try:
                    feature_dict[key] = float(value)
                except Exception:
                    feature_dict[key] = value
        
        st.success(f"成功提取 {len(feature_dict)} 个特征")
        return feature_dict
        
    except Exception as e:
        st.error(f"特征提取失败: {e}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def filter_and_standardize_features(extracted_features, background_features, standardization_params):
    """筛选影像组学特征，但不进行标准化"""
    st.session_state.all_extracted_features = extracted_features
    
    if standardization_params is not None:
        expected_features = standardization_params['continuous_features'] + standardization_params['binary_features']
        st.info(f"标准化参数期望 {len(expected_features)} 个特征")
        
        filtered_features = {}
        if background_features is not None:
            background_columns = list(background_features.columns)
            background_radiomic_features = []
            for col in background_columns:
                if any(keyword in col.lower() for keyword in ['original_', 'wavelet', 'log-sigma', 'squareroot_']):
                    background_radiomic_features.append(col)
            
            st.info(f"背景数据集包含 {len(background_radiomic_features)} 个影像组学特征")
            
            for feature in background_radiomic_features:
                if feature in extracted_features:
                    filtered_features[feature] = extracted_features[feature]
                else:
                    filtered_features[feature] = background_features[feature].median()
        
        st.success(f"从提取的特征中筛选了 {len(filtered_features)} 个用于显示的影像组学特征")
    else:
        filtered_features = {}
        for feature in RADIOMIC_FEATURES:
            if feature in extracted_features:
                filtered_features[feature] = extracted_features[feature]
            else:
                if background_features is not None and feature in background_features.columns:
                    filtered_features[feature] = background_features[feature].median()
                else:
                    filtered_features[feature] = 0.0
    
    return filtered_features

# 预处理函数
def preprocess_input(input_data_dict, feature_order_list, standardization_params=None):
    """将用户输入的字典转换为模型可接受的DataFrame，并进行预处理"""
    try:
        if standardization_params is None:
            st.error("⚠️ 标准化参数未加载，无法进行特征标准化！")
            return None
            
        continuous_features = standardization_params['continuous_features']
        mean_dict = standardization_params['mean_dict']
        std_dict = standardization_params['std_dict']
        
        # 收集所有可用的原始特征值
        all_raw_features = {}
        all_extracted = getattr(st.session_state, 'all_extracted_features', {})
        
        for feature, value in input_data_dict.items():
            all_raw_features[feature] = value
        
        for feature in st.session_state.radiomic_features:
            if feature not in all_raw_features:
                all_raw_features[feature] = st.session_state.radiomic_features[feature]
        
        for feature in all_extracted:
            if feature not in all_raw_features:
                all_raw_features[feature] = all_extracted[feature]
        
        # 构建最终用于预测的DataFrame
        final_features = {}
        standardization_log = []
        
        for feature in feature_order_list:
            if feature in mean_dict and feature in std_dict:
                if feature in all_raw_features:
                    raw_value = all_raw_features[feature]
                    standardized_value = (raw_value - mean_dict[feature]) / std_dict[feature]
                    final_features[feature] = standardized_value
                    standardization_log.append({
                        'feature': feature,
                        'raw_value': raw_value,
                        'mean': mean_dict[feature],
                        'std': std_dict[feature],
                        'standardized_value': standardized_value,
                        'status': '已标准化'
                    })
                else:
                    if shap_background_df is not None and feature in shap_background_df.columns:
                        raw_value = shap_background_df[feature].median()
                        standardized_value = (raw_value - mean_dict[feature]) / std_dict[feature]
                        final_features[feature] = standardized_value
                    else:
                        final_features[feature] = 0.0
            elif feature in all_raw_features:
                final_features[feature] = all_raw_features[feature]
                standardization_log.append({
                    'feature': feature,
                    'raw_value': all_raw_features[feature],
                    'mean': 'N/A',
                    'std': 'N/A',
                    'standardized_value': all_raw_features[feature],
                    'status': '二元特征'
                })
            else:
                if shap_background_df is not None and feature in shap_background_df.columns:
                    final_features[feature] = shap_background_df[feature].median()
                else:
                    final_features[feature] = 0.0
        
        # 存储标准化日志供后续显示
        st.session_state.standardization_log = standardization_log
        
        final_df = pd.DataFrame([final_features])
        final_df = final_df[feature_order_list]
        
        if final_df.isnull().values.any():
            nan_columns = final_df.columns[final_df.isnull().any()].tolist()
            st.error(f"❌ 以下特征包含NaN值: {nan_columns}")
            final_df.fillna(0, inplace=True)
        
        return final_df
            
    except Exception as e:
        st.error(f"❌ 预处理失败: {str(e)}")
        import traceback
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None

# --- 3. Streamlit 应用界面 ---
st.title("影像组学模型预测与SHAP可解释性分析")

# --- 侧边栏配置 ---
st.sidebar.header("模型选择")
model_files = [f for f in os.listdir(SERIALIZED_MODELS_DIR) if f.endswith(".joblib")]
if not model_files:
    st.error(f"在 '{SERIALIZED_MODELS_DIR}' 目录下未找到模型文件。")
    st.stop()

model_display_names = [os.path.splitext(f)[0].replace("_", " ") for f in model_files]
model_name_to_file_map = dict(zip(model_display_names, model_files))

selected_model_display_name = st.sidebar.selectbox(
    "选择预测模型:",
    options=model_display_names,
    index=model_display_names.index("Logistic Regression") if "Logistic Regression" in model_display_names else 0
)

selected_model_filename = model_name_to_file_map[selected_model_display_name]
model_path = os.path.join(SERIALIZED_MODELS_DIR, selected_model_filename)
model = load_model_file(model_path)

if model is None:
    st.error("模型加载失败。")
    st.stop()

st.sidebar.success(f"✓ 已加载模型: {selected_model_display_name}")

# --- 影像文件上传区域 ---
st.sidebar.header("影像文件上传")
st.sidebar.info("请上传NIfTI格式的原始图像和掩码文件（.nii或.nii.gz）")

image_file = st.sidebar.file_uploader(
    "选择原始图像文件:",
    type=['nii', 'gz'],
    key="image_uploader"
)

mask_file = st.sidebar.file_uploader(
    "选择掩码文件:",
    type=['nii', 'gz'],
    key="mask_uploader"
)

# 处理上传的文件并提取特征
if image_file and mask_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, image_file.name)
        mask_path = os.path.join(temp_dir, mask_file.name)
        
        with open(image_path, 'wb') as f:
            f.write(image_file.getbuffer())
        with open(mask_path, 'wb') as f:
            f.write(mask_file.getbuffer())
        
        if st.sidebar.button("提取影像组学特征", type="secondary"):
            with st.spinner("正在提取特征..."):
                extracted_features = extract_radiomics_features(
                    image_path, mask_path, radiomics_params
                )
                
                if extracted_features:
                    filtered_features = filter_and_standardize_features(
                        extracted_features, shap_background_df, standardization_info
                    )
                    
                    st.session_state.radiomic_features = filtered_features
                    st.sidebar.success(f"成功提取了 {len(filtered_features)} 个影像组学特征（原始值）")
                else:
                    st.sidebar.error("特征提取失败")

# --- 特征输入区域 ---
st.sidebar.header("特征值输入")

input_values = {}

default_input_values = {}
if shap_background_df is not None and not shap_background_df.empty:
    default_input_values = shap_background_df.iloc[0].to_dict()

# 影像组学特征输入
if RADIOMIC_FEATURES:
    st.sidebar.subheader("影像组学特征")
    radiomic_display = st.sidebar.checkbox("显示影像组学特征输入框", value=False)
    
    if radiomic_display:
        for feature in RADIOMIC_FEATURES:
            if feature in st.session_state.radiomic_features:
                default_val = st.session_state.radiomic_features[feature]
            else:
                default_val = default_input_values.get(feature, 0.0)
            
            input_values[feature] = st.sidebar.number_input(
                f"{feature}",
                value=float(default_val),
                format="%.6f",
                key=f"radiomic_{feature}"
            )
    else:
        for feature in RADIOMIC_FEATURES:
            if feature in st.session_state.radiomic_features:
                input_values[feature] = st.session_state.radiomic_features[feature]
            else:
                input_values[feature] = default_input_values.get(feature, 0.0)

# 临床特征输入
if CLINICAL_FEATURES:
    st.sidebar.subheader("临床特征（请手动输入原始值）")
    
    clinical_feature_types = {}
    if shap_background_df is not None:
        for feature in CLINICAL_FEATURES:
            if feature in shap_background_df.columns:
                var_type, unique_vals = identify_variable_type(shap_background_df[feature])
                clinical_feature_types[feature] = (var_type, unique_vals)
            else:
                clinical_feature_types[feature] = ('continuous', None)
    
    for feature in CLINICAL_FEATURES:
        default_val = default_input_values.get(feature, 0.0)
        var_type, unique_vals = clinical_feature_types.get(feature, ('continuous', None))
        
        if var_type == 'binary' and unique_vals is not None:
            if default_val not in unique_vals:
                default_val = unique_vals[0]
            
            input_values[feature] = st.sidebar.selectbox(
                f"{feature}",
                options=unique_vals,
                index=unique_vals.index(default_val) if default_val in unique_vals else 0,
                key=f"clinical_{feature}"
            )
        else:
            input_values[feature] = st.sidebar.number_input(
                f"{feature}",
                value=float(default_val) if isinstance(default_val, (int, float, np.number)) else 0.0,
                key=f"clinical_{feature}",
                format="%.2f"
            )

# --- 主界面 ---
if st.sidebar.button("执行预测与分析", type="primary", use_container_width=True):
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["特征预处理", "预测结果与SHAP分析", "标准化详情"])
    
    # 预处理数据
    processed_input_df = preprocess_input(input_values, FEATURE_ORDER, standardization_info)
    
    # Tab 1: 特征预处理
    with tab1:
        st.subheader("特征预处理概览")
        
        # 显示已标准化特征的摘要
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总特征数", len(FEATURE_ORDER))
        with col2:
            st.metric("连续特征数", len(standardization_info['continuous_features']) if standardization_info else 0)
        with col3:
            st.metric("二元特征数", len(standardization_info['binary_features']) if standardization_info else 0)
        
        # 简洁的特征展示
        with st.expander("查看标准化的影像组学特征", expanded=False):
            if RADIOMIC_FEATURES and hasattr(st.session_state, 'standardization_log'):
                radiomic_log = [log for log in st.session_state.standardization_log if log['feature'] in RADIOMIC_FEATURES and log['status'] == '已标准化']
                if radiomic_log:
                    for i, log in enumerate(radiomic_log[:5]):  # 只显示前5个
                        st.success(f"✓ 特征 '{log['feature']}' 已标准化: {log['raw_value']:.4f} → {log['standardized_value']:.4f}")
                    if len(radiomic_log) > 5:
                        st.info(f"... 还有 {len(radiomic_log)-5} 个特征已标准化")
        
        with st.expander("查看二元特征", expanded=False):
            binary_features_in_input = [f for f in input_values.keys() if f in standardization_info['binary_features']]
            if binary_features_in_input:
                for feature in binary_features_in_input:
                    st.info(f"✓ 特征 '{feature}' 使用原始值（二元特征）: {input_values[feature]}")
    
    # Tab 2: 预测结果与SHAP分析
    with tab2:
        if processed_input_df is not None:
            col_result, col_shap = st.columns([1, 2])
            
            # 预测结果列
            with col_result:
                st.subheader("模型预测结果")
                try:
                    model_type = str(type(model))
                    
                    if "XGBoost" in model_type or "xgboost" in model_type.lower():
                        try:
                            import xgboost as xgb
                            prediction_proba = model.predict_proba(processed_input_df)
                            prediction = model.predict(processed_input_df)
                        except Exception as xgb_error:
                            st.warning("尝试使用XGBoost DMatrix格式...")
                            dmatrix = xgb.DMatrix(processed_input_df)
                            prediction = model.predict(dmatrix)
                            prediction_proba = np.array([[1-prediction[0], prediction[0]]])
                            prediction = np.array([int(prediction[0] > 0.5)])
                    else:
                        prediction_proba = model.predict_proba(processed_input_df)
                        prediction = model.predict(processed_input_df)
                    
                    # 显示预测结果
                    st.metric(label="预测类别", value=str(prediction[0]))
                    
                    if len(prediction_proba[0]) > 1:
                        prob_positive_class = prediction_proba[0][1]
                    else:
                        prob_positive_class = prediction_proba[0][0]
                    
                    prob_positive_class = float(prob_positive_class)
                    
                    st.metric(label="阳性类别概率", value=f"{prob_positive_class:.4f}")
                    st.progress(prob_positive_class)
                    
                except Exception as e:
                    st.error(f"预测时发生错误: {e}")
                    prediction = None
            
            # SHAP分析列
            with col_shap:
                if prediction is not None:
                    st.subheader("SHAP 可解释性分析")
                    try:
                        explainer = None
                        model_type_str = str(type(model)).lower()
                        
                        shap_background_processed = shap_background_df[FEATURE_ORDER].copy()
                        
                        if "forest" in model_type_str or "tree" in model_type_str:
                            explainer = shap.TreeExplainer(model)
                        elif "logistic" in model_type_str or "linear" in model_type_str:
                            explainer = shap.LinearExplainer(model, shap_background_processed)
                        elif "xgboost" in model_type_str:
                            try:
                                explainer = shap.TreeExplainer(model)
                            except:
                                def xgb_predict(data):
                                    return model.predict_proba(data)[:, 1]
                                background_sample = shap_background_processed.sample(min(100, len(shap_background_processed)))
                                explainer = shap.KernelExplainer(xgb_predict, background_sample)
                        else:
                            def model_predict_proba_for_shap(data_as_np_array):
                                data_as_df = pd.DataFrame(data_as_np_array, columns=FEATURE_ORDER)
                                if hasattr(model, 'predict_proba'):
                                    return model.predict_proba(data_as_df)[:, 1]
                                else:
                                    return model.predict(data_as_df)
                            
                            background_sample = shap_background_processed.sample(min(100, len(shap_background_processed)))
                            explainer = shap.KernelExplainer(model_predict_proba_for_shap, background_sample)
                        
                        if explainer is not None:
                            shap_values_instance = explainer(processed_input_df)
                            
                            # 调整瀑布图大小
                            fig_waterfall = plt.figure(figsize=(12, 10))
                            
                            if isinstance(shap_values_instance, shap.Explanation):
                                shap.waterfall_plot(shap_values_instance[0], show=False, max_display=20)
                            else:
                                st.warning("SHAP值格式不支持瀑布图，尝试其他可视化...")
                                shap.summary_plot(shap_values_instance, processed_input_df, 
                                                feature_names=FEATURE_ORDER, show=False)
                            
                            plt.tight_layout(pad=2)
                            st.pyplot(fig_waterfall, clear_figure=True)
                            plt.close()
                        
                    except Exception as e:
                        st.error(f"SHAP分析时出错: {e}")
                        import traceback
                        with st.expander("查看详细错误信息"):
                            st.code(traceback.format_exc())
    
    # Tab 3: 标准化详情
    with tab3:
        st.subheader("特征标准化详细信息")
        
        if hasattr(st.session_state, 'standardization_log'):
            # 创建详细的标准化信息表格
            standardization_df = pd.DataFrame(st.session_state.standardization_log)
            
            # 分别显示连续特征和二元特征
            continuous_df = standardization_df[standardization_df['status'] == '已标准化']
            binary_df = standardization_df[standardization_df['status'] == '二元特征']
            
            if not continuous_df.empty:
                st.write("### 连续特征标准化详情")
                # 格式化数值列
                for col in ['raw_value', 'mean', 'std', 'standardized_value']:
                    if col in continuous_df.columns:
                        continuous_df[col] = continuous_df[col].apply(
                            lambda x: f"{float(x):.4f}" if isinstance(x, (int, float, np.number)) else x
                        )
                
                st.dataframe(
                    continuous_df[['feature', 'raw_value', 'mean', 'std', 'standardized_value']],
                    hide_index=True,
                    use_container_width=True
                )
            
            if not binary_df.empty:
                st.write("### 二元特征")
                st.dataframe(
                    binary_df[['feature', 'raw_value']],
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("请先执行预测以查看标准化详情。")

else:
    st.info("请上传影像文件或输入特征值，然后点击'执行预测与分析'按钮。")

# --- 应用说明 ---
st.sidebar.markdown("---")
st.sidebar.info(
    "**使用说明:**\n"
    "1. 选择预测模型\n"
    "2. 上传NIfTI格式的原始图像和掩码文件\n"
    "3. 点击'提取影像组学特征'自动填充影像特征\n"
    "4. 手动输入临床特征值\n"
    "5. 点击'执行预测与分析'查看结果"
)