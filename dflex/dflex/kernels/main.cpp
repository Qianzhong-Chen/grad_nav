#include <torch/extension.h>

#define CPU

#include "adjoint.h"

using namespace df;

template <typename T>
T cast(torch::Tensor t)
{{
    return (T)(t.data_ptr());
}}


float test_cpu_func(
	float var_c)
{
        //---------
    // primal vars
    const float var_0 = 1.0;
    const int var_1 = 2;
    float var_2;
    const float var_3 = 3.0;
    int var_4;
    bool var_5;
    const float var_6 = 2.0;
    float var_7;
    const float var_8 = 6.0;
    float var_9;
    //---------
    // forward
    var_2 = df::float(var_1);
    var_4 = df::int(var_3);
    df::print(var_2);
    df::print(var_4);
    var_5 = (var_c < var_3);
    if (var_5) {
    }
    var_7 = df::select(var_5, var_0, var_6);
    var_9 = df::mul(var_7, var_8);
    return var_9;
    

}

void adj_test_cpu_func(
	float var_c, 
	float & adj_c,
	float & adj_ret)
{
        //---------
    // primal vars
    const float var_0 = 1.0;
    const int var_1 = 2;
    float var_2;
    const float var_3 = 3.0;
    int var_4;
    bool var_5;
    const float var_6 = 2.0;
    float var_7;
    const float var_8 = 6.0;
    float var_9;
    //---------
    // dual vars
    float adj_0 = 0;
    int adj_1 = 0;
    float adj_2 = 0;
    float adj_3 = 0;
    int adj_4 = 0;
    bool adj_5 = 0;
    float adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    //---------
    // forward
    var_2 = df::float(var_1);
    var_4 = df::int(var_3);
    df::print(var_2);
    df::print(var_4);
    var_5 = (var_c < var_3);
    if (var_5) {
    }
    var_7 = df::select(var_5, var_0, var_6);
    var_9 = df::mul(var_7, var_8);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_9 += adj_ret;
    df::adj_mul(var_7, var_8, adj_7, adj_8, adj_9);
    df::adj_select(var_5, var_0, var_6, adj_5, adj_0, adj_6, adj_7);
    if (var_5) {
    }
    df::adj_print(var_4, adj_4);
    df::adj_print(var_2, adj_2);
    df::adj_int(var_3, adj_3, adj_4);
    df::adj_float(var_1, adj_1, adj_2);
    return;

}


df::float3 triangle_closest_point_barycentric_cpu_func(
	df::float3 var_a,
	df::float3 var_b,
	df::float3 var_c,
	df::float3 var_p)
{
        //---------
    // primal vars
    df::float3 var_0;
    df::float3 var_1;
    df::float3 var_2;
    float var_3;
    float var_4;
    const float var_5 = 0.0;
    bool var_6;
    bool var_7;
    bool var_8;
    const float var_9 = 1.0;
    df::float3 var_10;
    df::float3 var_11;
    float var_12;
    float var_13;
    bool var_14;
    bool var_15;
    bool var_16;
    df::float3 var_17;
    df::float3 var_18;
    float var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    bool var_24;
    bool var_25;
    bool var_26;
    bool var_27;
    float var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    float var_32;
    float var_33;
    bool var_34;
    bool var_35;
    bool var_36;
    df::float3 var_37;
    df::float3 var_38;
    float var_39;
    float var_40;
    float var_41;
    float var_42;
    float var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    bool var_47;
    float var_48;
    df::float3 var_49;
    df::float3 var_50;
    float var_51;
    float var_52;
    float var_53;
    float var_54;
    float var_55;
    float var_56;
    float var_57;
    float var_58;
    bool var_59;
    float var_60;
    bool var_61;
    float var_62;
    bool var_63;
    bool var_64;
    float var_65;
    df::float3 var_66;
    df::float3 var_67;
    float var_68;
    float var_69;
    float var_70;
    float var_71;
    float var_72;
    float var_73;
    float var_74;
    df::float3 var_75;
    //---------
    // forward
    var_0 = df::sub(var_b, var_a);
    var_1 = df::sub(var_c, var_a);
    var_2 = df::sub(var_p, var_a);
    var_3 = df::dot(var_0, var_2);
    var_4 = df::dot(var_1, var_2);
    var_6 = (var_3 <= var_5);
    var_7 = (var_4 <= var_5);
    var_8 = var_6 && var_7;
    if (var_8) {
    	var_10 = df::float3(var_9, var_5, var_5);
    	return var_10;
    }
    var_11 = df::sub(var_p, var_b);
    var_12 = df::dot(var_0, var_11);
    var_13 = df::dot(var_1, var_11);
    var_14 = (var_12 >= var_5);
    var_15 = (var_13 <= var_12);
    var_16 = var_14 && var_15;
    if (var_16) {
    	var_17 = df::float3(var_5, var_9, var_5);
    	return var_17;
    }
    var_18 = df::select(var_16, var_10, var_17);
    var_19 = df::mul(var_3, var_13);
    var_20 = df::mul(var_12, var_4);
    var_21 = df::sub(var_19, var_20);
    var_22 = df::sub(var_3, var_12);
    var_23 = df::div(var_3, var_22);
    var_24 = (var_21 <= var_5);
    var_25 = (var_3 >= var_5);
    var_26 = (var_12 <= var_5);
    var_27 = var_24 && var_25 && var_26;
    if (var_27) {
    	var_28 = df::sub(var_9, var_23);
    	var_29 = df::float3(var_28, var_23, var_5);
    	return var_29;
    }
    var_30 = df::select(var_27, var_18, var_29);
    var_31 = df::sub(var_p, var_c);
    var_32 = df::dot(var_0, var_31);
    var_33 = df::dot(var_1, var_31);
    var_34 = (var_33 >= var_5);
    var_35 = (var_32 <= var_33);
    var_36 = var_34 && var_35;
    if (var_36) {
    	var_37 = df::float3(var_5, var_5, var_9);
    	return var_37;
    }
    var_38 = df::select(var_36, var_30, var_37);
    var_39 = df::mul(var_32, var_4);
    var_40 = df::mul(var_3, var_33);
    var_41 = df::sub(var_39, var_40);
    var_42 = df::sub(var_4, var_33);
    var_43 = df::div(var_4, var_42);
    var_44 = (var_41 <= var_5);
    var_45 = (var_4 >= var_5);
    var_46 = (var_33 <= var_5);
    var_47 = var_44 && var_45 && var_46;
    if (var_47) {
    	var_48 = df::sub(var_9, var_43);
    	var_49 = df::float3(var_48, var_5, var_43);
    	return var_49;
    }
    var_50 = df::select(var_47, var_38, var_49);
    var_51 = df::mul(var_12, var_33);
    var_52 = df::mul(var_32, var_13);
    var_53 = df::sub(var_51, var_52);
    var_54 = df::sub(var_13, var_12);
    var_55 = df::sub(var_13, var_12);
    var_56 = df::sub(var_32, var_33);
    var_57 = df::add(var_55, var_56);
    var_58 = df::div(var_54, var_57);
    var_59 = (var_53 <= var_5);
    var_60 = df::sub(var_13, var_12);
    var_61 = (var_60 >= var_5);
    var_62 = df::sub(var_32, var_33);
    var_63 = (var_62 >= var_5);
    var_64 = var_59 && var_61 && var_63;
    if (var_64) {
    	var_65 = df::sub(var_9, var_58);
    	var_66 = df::float3(var_5, var_58, var_65);
    	return var_66;
    }
    var_67 = df::select(var_64, var_50, var_66);
    var_68 = df::add(var_53, var_41);
    var_69 = df::add(var_68, var_21);
    var_70 = df::div(var_9, var_69);
    var_71 = df::mul(var_41, var_70);
    var_72 = df::mul(var_21, var_70);
    var_73 = df::sub(var_9, var_71);
    var_74 = df::sub(var_73, var_72);
    var_75 = df::float3(var_74, var_71, var_72);
    return var_75;
    

}

void adj_triangle_closest_point_barycentric_cpu_func(
	df::float3 var_a,
	df::float3 var_b,
	df::float3 var_c,
	df::float3 var_p, 
	df::float3 & adj_a,
	df::float3 & adj_b,
	df::float3 & adj_c,
	df::float3 & adj_p,
	df::float3 & adj_ret)
{
        //---------
    // primal vars
    df::float3 var_0;
    df::float3 var_1;
    df::float3 var_2;
    float var_3;
    float var_4;
    const float var_5 = 0.0;
    bool var_6;
    bool var_7;
    bool var_8;
    const float var_9 = 1.0;
    df::float3 var_10;
    df::float3 var_11;
    float var_12;
    float var_13;
    bool var_14;
    bool var_15;
    bool var_16;
    df::float3 var_17;
    df::float3 var_18;
    float var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    bool var_24;
    bool var_25;
    bool var_26;
    bool var_27;
    float var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    float var_32;
    float var_33;
    bool var_34;
    bool var_35;
    bool var_36;
    df::float3 var_37;
    df::float3 var_38;
    float var_39;
    float var_40;
    float var_41;
    float var_42;
    float var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    bool var_47;
    float var_48;
    df::float3 var_49;
    df::float3 var_50;
    float var_51;
    float var_52;
    float var_53;
    float var_54;
    float var_55;
    float var_56;
    float var_57;
    float var_58;
    bool var_59;
    float var_60;
    bool var_61;
    float var_62;
    bool var_63;
    bool var_64;
    float var_65;
    df::float3 var_66;
    df::float3 var_67;
    float var_68;
    float var_69;
    float var_70;
    float var_71;
    float var_72;
    float var_73;
    float var_74;
    df::float3 var_75;
    //---------
    // dual vars
    df::float3 adj_0 = 0;
    df::float3 adj_1 = 0;
    df::float3 adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    float adj_5 = 0;
    bool adj_6 = 0;
    bool adj_7 = 0;
    bool adj_8 = 0;
    float adj_9 = 0;
    df::float3 adj_10 = 0;
    df::float3 adj_11 = 0;
    float adj_12 = 0;
    float adj_13 = 0;
    bool adj_14 = 0;
    bool adj_15 = 0;
    bool adj_16 = 0;
    df::float3 adj_17 = 0;
    df::float3 adj_18 = 0;
    float adj_19 = 0;
    float adj_20 = 0;
    float adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    bool adj_24 = 0;
    bool adj_25 = 0;
    bool adj_26 = 0;
    bool adj_27 = 0;
    float adj_28 = 0;
    df::float3 adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    float adj_32 = 0;
    float adj_33 = 0;
    bool adj_34 = 0;
    bool adj_35 = 0;
    bool adj_36 = 0;
    df::float3 adj_37 = 0;
    df::float3 adj_38 = 0;
    float adj_39 = 0;
    float adj_40 = 0;
    float adj_41 = 0;
    float adj_42 = 0;
    float adj_43 = 0;
    bool adj_44 = 0;
    bool adj_45 = 0;
    bool adj_46 = 0;
    bool adj_47 = 0;
    float adj_48 = 0;
    df::float3 adj_49 = 0;
    df::float3 adj_50 = 0;
    float adj_51 = 0;
    float adj_52 = 0;
    float adj_53 = 0;
    float adj_54 = 0;
    float adj_55 = 0;
    float adj_56 = 0;
    float adj_57 = 0;
    float adj_58 = 0;
    bool adj_59 = 0;
    float adj_60 = 0;
    bool adj_61 = 0;
    float adj_62 = 0;
    bool adj_63 = 0;
    bool adj_64 = 0;
    float adj_65 = 0;
    df::float3 adj_66 = 0;
    df::float3 adj_67 = 0;
    float adj_68 = 0;
    float adj_69 = 0;
    float adj_70 = 0;
    float adj_71 = 0;
    float adj_72 = 0;
    float adj_73 = 0;
    float adj_74 = 0;
    df::float3 adj_75 = 0;
    //---------
    // forward
    var_0 = df::sub(var_b, var_a);
    var_1 = df::sub(var_c, var_a);
    var_2 = df::sub(var_p, var_a);
    var_3 = df::dot(var_0, var_2);
    var_4 = df::dot(var_1, var_2);
    var_6 = (var_3 <= var_5);
    var_7 = (var_4 <= var_5);
    var_8 = var_6 && var_7;
    if (var_8) {
    	var_10 = df::float3(var_9, var_5, var_5);
    	goto label0;
    }
    var_11 = df::sub(var_p, var_b);
    var_12 = df::dot(var_0, var_11);
    var_13 = df::dot(var_1, var_11);
    var_14 = (var_12 >= var_5);
    var_15 = (var_13 <= var_12);
    var_16 = var_14 && var_15;
    if (var_16) {
    	var_17 = df::float3(var_5, var_9, var_5);
    	goto label1;
    }
    var_18 = df::select(var_16, var_10, var_17);
    var_19 = df::mul(var_3, var_13);
    var_20 = df::mul(var_12, var_4);
    var_21 = df::sub(var_19, var_20);
    var_22 = df::sub(var_3, var_12);
    var_23 = df::div(var_3, var_22);
    var_24 = (var_21 <= var_5);
    var_25 = (var_3 >= var_5);
    var_26 = (var_12 <= var_5);
    var_27 = var_24 && var_25 && var_26;
    if (var_27) {
    	var_28 = df::sub(var_9, var_23);
    	var_29 = df::float3(var_28, var_23, var_5);
    	goto label2;
    }
    var_30 = df::select(var_27, var_18, var_29);
    var_31 = df::sub(var_p, var_c);
    var_32 = df::dot(var_0, var_31);
    var_33 = df::dot(var_1, var_31);
    var_34 = (var_33 >= var_5);
    var_35 = (var_32 <= var_33);
    var_36 = var_34 && var_35;
    if (var_36) {
    	var_37 = df::float3(var_5, var_5, var_9);
    	goto label3;
    }
    var_38 = df::select(var_36, var_30, var_37);
    var_39 = df::mul(var_32, var_4);
    var_40 = df::mul(var_3, var_33);
    var_41 = df::sub(var_39, var_40);
    var_42 = df::sub(var_4, var_33);
    var_43 = df::div(var_4, var_42);
    var_44 = (var_41 <= var_5);
    var_45 = (var_4 >= var_5);
    var_46 = (var_33 <= var_5);
    var_47 = var_44 && var_45 && var_46;
    if (var_47) {
    	var_48 = df::sub(var_9, var_43);
    	var_49 = df::float3(var_48, var_5, var_43);
    	goto label4;
    }
    var_50 = df::select(var_47, var_38, var_49);
    var_51 = df::mul(var_12, var_33);
    var_52 = df::mul(var_32, var_13);
    var_53 = df::sub(var_51, var_52);
    var_54 = df::sub(var_13, var_12);
    var_55 = df::sub(var_13, var_12);
    var_56 = df::sub(var_32, var_33);
    var_57 = df::add(var_55, var_56);
    var_58 = df::div(var_54, var_57);
    var_59 = (var_53 <= var_5);
    var_60 = df::sub(var_13, var_12);
    var_61 = (var_60 >= var_5);
    var_62 = df::sub(var_32, var_33);
    var_63 = (var_62 >= var_5);
    var_64 = var_59 && var_61 && var_63;
    if (var_64) {
    	var_65 = df::sub(var_9, var_58);
    	var_66 = df::float3(var_5, var_58, var_65);
    	goto label5;
    }
    var_67 = df::select(var_64, var_50, var_66);
    var_68 = df::add(var_53, var_41);
    var_69 = df::add(var_68, var_21);
    var_70 = df::div(var_9, var_69);
    var_71 = df::mul(var_41, var_70);
    var_72 = df::mul(var_21, var_70);
    var_73 = df::sub(var_9, var_71);
    var_74 = df::sub(var_73, var_72);
    var_75 = df::float3(var_74, var_71, var_72);
    goto label6;
    //---------
    // reverse
    label6:;
    adj_75 += adj_ret;
    df::adj_float3(var_74, var_71, var_72, adj_74, adj_71, adj_72, adj_75);
    df::adj_sub(var_73, var_72, adj_73, adj_72, adj_74);
    df::adj_sub(var_9, var_71, adj_9, adj_71, adj_73);
    df::adj_mul(var_21, var_70, adj_21, adj_70, adj_72);
    df::adj_mul(var_41, var_70, adj_41, adj_70, adj_71);
    df::adj_div(var_9, var_69, adj_9, adj_69, adj_70);
    df::adj_add(var_68, var_21, adj_68, adj_21, adj_69);
    df::adj_add(var_53, var_41, adj_53, adj_41, adj_68);
    df::adj_select(var_64, var_50, var_66, adj_64, adj_50, adj_66, adj_67);
    if (var_64) {
    	label5:;
    	adj_66 += adj_ret;
    	df::adj_float3(var_5, var_58, var_65, adj_5, adj_58, adj_65, adj_66);
    	df::adj_sub(var_9, var_58, adj_9, adj_58, adj_65);
    }
    df::adj_sub(var_32, var_33, adj_32, adj_33, adj_62);
    df::adj_sub(var_13, var_12, adj_13, adj_12, adj_60);
    df::adj_div(var_54, var_57, adj_54, adj_57, adj_58);
    df::adj_add(var_55, var_56, adj_55, adj_56, adj_57);
    df::adj_sub(var_32, var_33, adj_32, adj_33, adj_56);
    df::adj_sub(var_13, var_12, adj_13, adj_12, adj_55);
    df::adj_sub(var_13, var_12, adj_13, adj_12, adj_54);
    df::adj_sub(var_51, var_52, adj_51, adj_52, adj_53);
    df::adj_mul(var_32, var_13, adj_32, adj_13, adj_52);
    df::adj_mul(var_12, var_33, adj_12, adj_33, adj_51);
    df::adj_select(var_47, var_38, var_49, adj_47, adj_38, adj_49, adj_50);
    if (var_47) {
    	label4:;
    	adj_49 += adj_ret;
    	df::adj_float3(var_48, var_5, var_43, adj_48, adj_5, adj_43, adj_49);
    	df::adj_sub(var_9, var_43, adj_9, adj_43, adj_48);
    }
    df::adj_div(var_4, var_42, adj_4, adj_42, adj_43);
    df::adj_sub(var_4, var_33, adj_4, adj_33, adj_42);
    df::adj_sub(var_39, var_40, adj_39, adj_40, adj_41);
    df::adj_mul(var_3, var_33, adj_3, adj_33, adj_40);
    df::adj_mul(var_32, var_4, adj_32, adj_4, adj_39);
    df::adj_select(var_36, var_30, var_37, adj_36, adj_30, adj_37, adj_38);
    if (var_36) {
    	label3:;
    	adj_37 += adj_ret;
    	df::adj_float3(var_5, var_5, var_9, adj_5, adj_5, adj_9, adj_37);
    }
    df::adj_dot(var_1, var_31, adj_1, adj_31, adj_33);
    df::adj_dot(var_0, var_31, adj_0, adj_31, adj_32);
    df::adj_sub(var_p, var_c, adj_p, adj_c, adj_31);
    df::adj_select(var_27, var_18, var_29, adj_27, adj_18, adj_29, adj_30);
    if (var_27) {
    	label2:;
    	adj_29 += adj_ret;
    	df::adj_float3(var_28, var_23, var_5, adj_28, adj_23, adj_5, adj_29);
    	df::adj_sub(var_9, var_23, adj_9, adj_23, adj_28);
    }
    df::adj_div(var_3, var_22, adj_3, adj_22, adj_23);
    df::adj_sub(var_3, var_12, adj_3, adj_12, adj_22);
    df::adj_sub(var_19, var_20, adj_19, adj_20, adj_21);
    df::adj_mul(var_12, var_4, adj_12, adj_4, adj_20);
    df::adj_mul(var_3, var_13, adj_3, adj_13, adj_19);
    df::adj_select(var_16, var_10, var_17, adj_16, adj_10, adj_17, adj_18);
    if (var_16) {
    	label1:;
    	adj_17 += adj_ret;
    	df::adj_float3(var_5, var_9, var_5, adj_5, adj_9, adj_5, adj_17);
    }
    df::adj_dot(var_1, var_11, adj_1, adj_11, adj_13);
    df::adj_dot(var_0, var_11, adj_0, adj_11, adj_12);
    df::adj_sub(var_p, var_b, adj_p, adj_b, adj_11);
    if (var_8) {
    	label0:;
    	adj_10 += adj_ret;
    	df::adj_float3(var_9, var_5, var_5, adj_9, adj_5, adj_5, adj_10);
    }
    df::adj_dot(var_1, var_2, adj_1, adj_2, adj_4);
    df::adj_dot(var_0, var_2, adj_0, adj_2, adj_3);
    df::adj_sub(var_p, var_a, adj_p, adj_a, adj_2);
    df::adj_sub(var_c, var_a, adj_c, adj_a, adj_1);
    df::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    return;

}


float sphere_sdf_cpu_func(
	df::float3 var_center,
	float var_radius,
	df::float3 var_p)
{
        //---------
    // primal vars
    df::float3 var_0;
    float var_1;
    float var_2;
    //---------
    // forward
    var_0 = df::sub(var_p, var_center);
    var_1 = df::length(var_0);
    var_2 = df::sub(var_1, var_radius);
    return var_2;
    

}

void adj_sphere_sdf_cpu_func(
	df::float3 var_center,
	float var_radius,
	df::float3 var_p, 
	df::float3 & adj_center,
	float & adj_radius,
	df::float3 & adj_p,
	float & adj_ret)
{
        //---------
    // primal vars
    df::float3 var_0;
    float var_1;
    float var_2;
    //---------
    // dual vars
    df::float3 adj_0 = 0;
    float adj_1 = 0;
    float adj_2 = 0;
    //---------
    // forward
    var_0 = df::sub(var_p, var_center);
    var_1 = df::length(var_0);
    var_2 = df::sub(var_1, var_radius);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_2 += adj_ret;
    df::adj_sub(var_1, var_radius, adj_1, adj_radius, adj_2);
    df::adj_length(var_0, adj_0, adj_1);
    df::adj_sub(var_p, var_center, adj_p, adj_center, adj_0);
    return;

}


df::float3 sphere_sdf_grad_cpu_func(
	df::float3 var_center,
	float var_radius,
	df::float3 var_p)
{
        //---------
    // primal vars
    df::float3 var_0;
    df::float3 var_1;
    //---------
    // forward
    var_0 = df::sub(var_p, var_center);
    var_1 = df::normalize(var_0);
    return var_1;
    

}

void adj_sphere_sdf_grad_cpu_func(
	df::float3 var_center,
	float var_radius,
	df::float3 var_p, 
	df::float3 & adj_center,
	float & adj_radius,
	df::float3 & adj_p,
	df::float3 & adj_ret)
{
        //---------
    // primal vars
    df::float3 var_0;
    df::float3 var_1;
    //---------
    // dual vars
    df::float3 adj_0 = 0;
    df::float3 adj_1 = 0;
    //---------
    // forward
    var_0 = df::sub(var_p, var_center);
    var_1 = df::normalize(var_0);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_1 += adj_ret;
    df::adj_normalize(var_0, adj_0, adj_1);
    df::adj_sub(var_p, var_center, adj_p, adj_center, adj_0);
    return;

}


float box_sdf_cpu_func(
	df::float3 var_upper,
	df::float3 var_p)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    float var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    const int var_10 = 2;
    float var_11;
    float var_12;
    float var_13;
    float var_14;
    const float var_15 = 0.0;
    float var_16;
    float var_17;
    float var_18;
    df::float3 var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = df::abs(var_1);
    var_3 = df::index(var_upper, var_0);
    var_4 = df::sub(var_2, var_3);
    var_6 = df::index(var_p, var_5);
    var_7 = df::abs(var_6);
    var_8 = df::index(var_upper, var_5);
    var_9 = df::sub(var_7, var_8);
    var_11 = df::index(var_p, var_10);
    var_12 = df::abs(var_11);
    var_13 = df::index(var_upper, var_10);
    var_14 = df::sub(var_12, var_13);
    var_16 = df::max(var_4, var_15);
    var_17 = df::max(var_9, var_15);
    var_18 = df::max(var_14, var_15);
    var_19 = df::float3(var_16, var_17, var_18);
    var_20 = df::length(var_19);
    var_21 = df::max(var_9, var_14);
    var_22 = df::max(var_4, var_21);
    var_23 = df::min(var_22, var_15);
    var_24 = df::add(var_20, var_23);
    return var_24;
    

}

void adj_box_sdf_cpu_func(
	df::float3 var_upper,
	df::float3 var_p, 
	df::float3 & adj_upper,
	df::float3 & adj_p,
	float & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    float var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    const int var_10 = 2;
    float var_11;
    float var_12;
    float var_13;
    float var_14;
    const float var_15 = 0.0;
    float var_16;
    float var_17;
    float var_18;
    df::float3 var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    //---------
    // dual vars
    int adj_0 = 0;
    float adj_1 = 0;
    float adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    int adj_5 = 0;
    float adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    int adj_10 = 0;
    float adj_11 = 0;
    float adj_12 = 0;
    float adj_13 = 0;
    float adj_14 = 0;
    float adj_15 = 0;
    float adj_16 = 0;
    float adj_17 = 0;
    float adj_18 = 0;
    df::float3 adj_19 = 0;
    float adj_20 = 0;
    float adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = df::abs(var_1);
    var_3 = df::index(var_upper, var_0);
    var_4 = df::sub(var_2, var_3);
    var_6 = df::index(var_p, var_5);
    var_7 = df::abs(var_6);
    var_8 = df::index(var_upper, var_5);
    var_9 = df::sub(var_7, var_8);
    var_11 = df::index(var_p, var_10);
    var_12 = df::abs(var_11);
    var_13 = df::index(var_upper, var_10);
    var_14 = df::sub(var_12, var_13);
    var_16 = df::max(var_4, var_15);
    var_17 = df::max(var_9, var_15);
    var_18 = df::max(var_14, var_15);
    var_19 = df::float3(var_16, var_17, var_18);
    var_20 = df::length(var_19);
    var_21 = df::max(var_9, var_14);
    var_22 = df::max(var_4, var_21);
    var_23 = df::min(var_22, var_15);
    var_24 = df::add(var_20, var_23);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_24 += adj_ret;
    df::adj_add(var_20, var_23, adj_20, adj_23, adj_24);
    df::adj_min(var_22, var_15, adj_22, adj_15, adj_23);
    df::adj_max(var_4, var_21, adj_4, adj_21, adj_22);
    df::adj_max(var_9, var_14, adj_9, adj_14, adj_21);
    df::adj_length(var_19, adj_19, adj_20);
    df::adj_float3(var_16, var_17, var_18, adj_16, adj_17, adj_18, adj_19);
    df::adj_max(var_14, var_15, adj_14, adj_15, adj_18);
    df::adj_max(var_9, var_15, adj_9, adj_15, adj_17);
    df::adj_max(var_4, var_15, adj_4, adj_15, adj_16);
    df::adj_sub(var_12, var_13, adj_12, adj_13, adj_14);
    df::adj_index(var_upper, var_10, adj_upper, adj_10, adj_13);
    df::adj_abs(var_11, adj_11, adj_12);
    df::adj_index(var_p, var_10, adj_p, adj_10, adj_11);
    df::adj_sub(var_7, var_8, adj_7, adj_8, adj_9);
    df::adj_index(var_upper, var_5, adj_upper, adj_5, adj_8);
    df::adj_abs(var_6, adj_6, adj_7);
    df::adj_index(var_p, var_5, adj_p, adj_5, adj_6);
    df::adj_sub(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_index(var_upper, var_0, adj_upper, adj_0, adj_3);
    df::adj_abs(var_1, adj_1, adj_2);
    df::adj_index(var_p, var_0, adj_p, adj_0, adj_1);
    return;

}


df::float3 box_sdf_grad_cpu_func(
	df::float3 var_upper,
	df::float3 var_p)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    float var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    const int var_10 = 2;
    float var_11;
    float var_12;
    float var_13;
    float var_14;
    const float var_15 = 0.0;
    bool var_16;
    bool var_17;
    bool var_18;
    bool var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    float var_32;
    float var_33;
    float var_34;
    df::float3 var_35;
    df::float3 var_36;
    df::float3 var_37;
    float var_38;
    float var_39;
    float var_40;
    float var_41;
    float var_42;
    float var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    df::float3 var_47;
    df::float3 var_48;
    bool var_49;
    bool var_50;
    bool var_51;
    df::float3 var_52;
    df::float3 var_53;
    bool var_54;
    bool var_55;
    bool var_56;
    df::float3 var_57;
    df::float3 var_58;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = df::abs(var_1);
    var_3 = df::index(var_upper, var_0);
    var_4 = df::sub(var_2, var_3);
    var_6 = df::index(var_p, var_5);
    var_7 = df::abs(var_6);
    var_8 = df::index(var_upper, var_5);
    var_9 = df::sub(var_7, var_8);
    var_11 = df::index(var_p, var_10);
    var_12 = df::abs(var_11);
    var_13 = df::index(var_upper, var_10);
    var_14 = df::sub(var_12, var_13);
    var_16 = (var_4 > var_15);
    var_17 = (var_9 > var_15);
    var_18 = (var_14 > var_15);
    var_19 = var_16 || var_17 || var_18;
    if (var_19) {
    	var_20 = df::index(var_p, var_0);
    	var_21 = df::index(var_upper, var_0);
    	var_22 = df::sub(var_15, var_21);
    	var_23 = df::index(var_upper, var_0);
    	var_24 = df::clamp(var_20, var_22, var_23);
    	var_25 = df::index(var_p, var_5);
    	var_26 = df::index(var_upper, var_5);
    	var_27 = df::sub(var_15, var_26);
    	var_28 = df::index(var_upper, var_5);
    	var_29 = df::clamp(var_25, var_27, var_28);
    	var_30 = df::index(var_p, var_10);
    	var_31 = df::index(var_upper, var_10);
    	var_32 = df::sub(var_15, var_31);
    	var_33 = df::index(var_upper, var_10);
    	var_34 = df::clamp(var_30, var_32, var_33);
    	var_35 = df::float3(var_24, var_29, var_34);
    	var_36 = df::sub(var_p, var_35);
    	var_37 = df::normalize(var_36);
    	return var_37;
    }
    var_38 = df::index(var_p, var_0);
    var_39 = df::sign(var_38);
    var_40 = df::index(var_p, var_5);
    var_41 = df::sign(var_40);
    var_42 = df::index(var_p, var_10);
    var_43 = df::sign(var_42);
    var_44 = (var_4 > var_9);
    var_45 = (var_4 > var_14);
    var_46 = var_44 && var_45;
    if (var_46) {
    	var_47 = df::float3(var_39, var_15, var_15);
    	return var_47;
    }
    var_48 = df::select(var_46, var_37, var_47);
    var_49 = (var_9 > var_4);
    var_50 = (var_9 > var_14);
    var_51 = var_49 && var_50;
    if (var_51) {
    	var_52 = df::float3(var_15, var_41, var_15);
    	return var_52;
    }
    var_53 = df::select(var_51, var_48, var_52);
    var_54 = (var_14 > var_4);
    var_55 = (var_14 > var_9);
    var_56 = var_54 && var_55;
    if (var_56) {
    	var_57 = df::float3(var_15, var_15, var_43);
    	return var_57;
    }
    var_58 = df::select(var_56, var_53, var_57);
    

}

void adj_box_sdf_grad_cpu_func(
	df::float3 var_upper,
	df::float3 var_p, 
	df::float3 & adj_upper,
	df::float3 & adj_p,
	df::float3 & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    float var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    const int var_10 = 2;
    float var_11;
    float var_12;
    float var_13;
    float var_14;
    const float var_15 = 0.0;
    bool var_16;
    bool var_17;
    bool var_18;
    bool var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    float var_32;
    float var_33;
    float var_34;
    df::float3 var_35;
    df::float3 var_36;
    df::float3 var_37;
    float var_38;
    float var_39;
    float var_40;
    float var_41;
    float var_42;
    float var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    df::float3 var_47;
    df::float3 var_48;
    bool var_49;
    bool var_50;
    bool var_51;
    df::float3 var_52;
    df::float3 var_53;
    bool var_54;
    bool var_55;
    bool var_56;
    df::float3 var_57;
    df::float3 var_58;
    //---------
    // dual vars
    int adj_0 = 0;
    float adj_1 = 0;
    float adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    int adj_5 = 0;
    float adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    int adj_10 = 0;
    float adj_11 = 0;
    float adj_12 = 0;
    float adj_13 = 0;
    float adj_14 = 0;
    float adj_15 = 0;
    bool adj_16 = 0;
    bool adj_17 = 0;
    bool adj_18 = 0;
    bool adj_19 = 0;
    float adj_20 = 0;
    float adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    float adj_26 = 0;
    float adj_27 = 0;
    float adj_28 = 0;
    float adj_29 = 0;
    float adj_30 = 0;
    float adj_31 = 0;
    float adj_32 = 0;
    float adj_33 = 0;
    float adj_34 = 0;
    df::float3 adj_35 = 0;
    df::float3 adj_36 = 0;
    df::float3 adj_37 = 0;
    float adj_38 = 0;
    float adj_39 = 0;
    float adj_40 = 0;
    float adj_41 = 0;
    float adj_42 = 0;
    float adj_43 = 0;
    bool adj_44 = 0;
    bool adj_45 = 0;
    bool adj_46 = 0;
    df::float3 adj_47 = 0;
    df::float3 adj_48 = 0;
    bool adj_49 = 0;
    bool adj_50 = 0;
    bool adj_51 = 0;
    df::float3 adj_52 = 0;
    df::float3 adj_53 = 0;
    bool adj_54 = 0;
    bool adj_55 = 0;
    bool adj_56 = 0;
    df::float3 adj_57 = 0;
    df::float3 adj_58 = 0;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = df::abs(var_1);
    var_3 = df::index(var_upper, var_0);
    var_4 = df::sub(var_2, var_3);
    var_6 = df::index(var_p, var_5);
    var_7 = df::abs(var_6);
    var_8 = df::index(var_upper, var_5);
    var_9 = df::sub(var_7, var_8);
    var_11 = df::index(var_p, var_10);
    var_12 = df::abs(var_11);
    var_13 = df::index(var_upper, var_10);
    var_14 = df::sub(var_12, var_13);
    var_16 = (var_4 > var_15);
    var_17 = (var_9 > var_15);
    var_18 = (var_14 > var_15);
    var_19 = var_16 || var_17 || var_18;
    if (var_19) {
    	var_20 = df::index(var_p, var_0);
    	var_21 = df::index(var_upper, var_0);
    	var_22 = df::sub(var_15, var_21);
    	var_23 = df::index(var_upper, var_0);
    	var_24 = df::clamp(var_20, var_22, var_23);
    	var_25 = df::index(var_p, var_5);
    	var_26 = df::index(var_upper, var_5);
    	var_27 = df::sub(var_15, var_26);
    	var_28 = df::index(var_upper, var_5);
    	var_29 = df::clamp(var_25, var_27, var_28);
    	var_30 = df::index(var_p, var_10);
    	var_31 = df::index(var_upper, var_10);
    	var_32 = df::sub(var_15, var_31);
    	var_33 = df::index(var_upper, var_10);
    	var_34 = df::clamp(var_30, var_32, var_33);
    	var_35 = df::float3(var_24, var_29, var_34);
    	var_36 = df::sub(var_p, var_35);
    	var_37 = df::normalize(var_36);
    	goto label0;
    }
    var_38 = df::index(var_p, var_0);
    var_39 = df::sign(var_38);
    var_40 = df::index(var_p, var_5);
    var_41 = df::sign(var_40);
    var_42 = df::index(var_p, var_10);
    var_43 = df::sign(var_42);
    var_44 = (var_4 > var_9);
    var_45 = (var_4 > var_14);
    var_46 = var_44 && var_45;
    if (var_46) {
    	var_47 = df::float3(var_39, var_15, var_15);
    	goto label1;
    }
    var_48 = df::select(var_46, var_37, var_47);
    var_49 = (var_9 > var_4);
    var_50 = (var_9 > var_14);
    var_51 = var_49 && var_50;
    if (var_51) {
    	var_52 = df::float3(var_15, var_41, var_15);
    	goto label2;
    }
    var_53 = df::select(var_51, var_48, var_52);
    var_54 = (var_14 > var_4);
    var_55 = (var_14 > var_9);
    var_56 = var_54 && var_55;
    if (var_56) {
    	var_57 = df::float3(var_15, var_15, var_43);
    	goto label3;
    }
    var_58 = df::select(var_56, var_53, var_57);
    //---------
    // reverse
    df::adj_select(var_56, var_53, var_57, adj_56, adj_53, adj_57, adj_58);
    if (var_56) {
    	label3:;
    	adj_57 += adj_ret;
    	df::adj_float3(var_15, var_15, var_43, adj_15, adj_15, adj_43, adj_57);
    }
    df::adj_select(var_51, var_48, var_52, adj_51, adj_48, adj_52, adj_53);
    if (var_51) {
    	label2:;
    	adj_52 += adj_ret;
    	df::adj_float3(var_15, var_41, var_15, adj_15, adj_41, adj_15, adj_52);
    }
    df::adj_select(var_46, var_37, var_47, adj_46, adj_37, adj_47, adj_48);
    if (var_46) {
    	label1:;
    	adj_47 += adj_ret;
    	df::adj_float3(var_39, var_15, var_15, adj_39, adj_15, adj_15, adj_47);
    }
    df::adj_sign(var_42, adj_42, adj_43);
    df::adj_index(var_p, var_10, adj_p, adj_10, adj_42);
    df::adj_sign(var_40, adj_40, adj_41);
    df::adj_index(var_p, var_5, adj_p, adj_5, adj_40);
    df::adj_sign(var_38, adj_38, adj_39);
    df::adj_index(var_p, var_0, adj_p, adj_0, adj_38);
    if (var_19) {
    	label0:;
    	adj_37 += adj_ret;
    	df::adj_normalize(var_36, adj_36, adj_37);
    	df::adj_sub(var_p, var_35, adj_p, adj_35, adj_36);
    	df::adj_float3(var_24, var_29, var_34, adj_24, adj_29, adj_34, adj_35);
    	df::adj_clamp(var_30, var_32, var_33, adj_30, adj_32, adj_33, adj_34);
    	df::adj_index(var_upper, var_10, adj_upper, adj_10, adj_33);
    	df::adj_sub(var_15, var_31, adj_15, adj_31, adj_32);
    	df::adj_index(var_upper, var_10, adj_upper, adj_10, adj_31);
    	df::adj_index(var_p, var_10, adj_p, adj_10, adj_30);
    	df::adj_clamp(var_25, var_27, var_28, adj_25, adj_27, adj_28, adj_29);
    	df::adj_index(var_upper, var_5, adj_upper, adj_5, adj_28);
    	df::adj_sub(var_15, var_26, adj_15, adj_26, adj_27);
    	df::adj_index(var_upper, var_5, adj_upper, adj_5, adj_26);
    	df::adj_index(var_p, var_5, adj_p, adj_5, adj_25);
    	df::adj_clamp(var_20, var_22, var_23, adj_20, adj_22, adj_23, adj_24);
    	df::adj_index(var_upper, var_0, adj_upper, adj_0, adj_23);
    	df::adj_sub(var_15, var_21, adj_15, adj_21, adj_22);
    	df::adj_index(var_upper, var_0, adj_upper, adj_0, adj_21);
    	df::adj_index(var_p, var_0, adj_p, adj_0, adj_20);
    }
    df::adj_sub(var_12, var_13, adj_12, adj_13, adj_14);
    df::adj_index(var_upper, var_10, adj_upper, adj_10, adj_13);
    df::adj_abs(var_11, adj_11, adj_12);
    df::adj_index(var_p, var_10, adj_p, adj_10, adj_11);
    df::adj_sub(var_7, var_8, adj_7, adj_8, adj_9);
    df::adj_index(var_upper, var_5, adj_upper, adj_5, adj_8);
    df::adj_abs(var_6, adj_6, adj_7);
    df::adj_index(var_p, var_5, adj_p, adj_5, adj_6);
    df::adj_sub(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_index(var_upper, var_0, adj_upper, adj_0, adj_3);
    df::adj_abs(var_1, adj_1, adj_2);
    df::adj_index(var_p, var_0, adj_p, adj_0, adj_1);
    return;

}


float capsule_sdf_cpu_func(
	float var_radius,
	float var_half_width,
	df::float3 var_p)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    bool var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    const int var_7 = 2;
    float var_8;
    df::float3 var_9;
    float var_10;
    float var_11;
    float var_12;
    const float var_13 = 0.0;
    float var_14;
    bool var_15;
    float var_16;
    float var_17;
    float var_18;
    float var_19;
    df::float3 var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    df::float3 var_26;
    float var_27;
    float var_28;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = (var_1 > var_half_width);
    if (var_2) {
    	var_3 = df::index(var_p, var_0);
    	var_4 = df::sub(var_3, var_half_width);
    	var_6 = df::index(var_p, var_5);
    	var_8 = df::index(var_p, var_7);
    	var_9 = df::float3(var_4, var_6, var_8);
    	var_10 = df::length(var_9);
    	var_11 = df::sub(var_10, var_radius);
    	return var_11;
    }
    var_12 = df::index(var_p, var_0);
    var_14 = df::sub(var_13, var_half_width);
    var_15 = (var_12 < var_14);
    if (var_15) {
    	var_16 = df::index(var_p, var_0);
    	var_17 = df::add(var_16, var_half_width);
    	var_18 = df::index(var_p, var_5);
    	var_19 = df::index(var_p, var_7);
    	var_20 = df::float3(var_17, var_18, var_19);
    	var_21 = df::length(var_20);
    	var_22 = df::sub(var_21, var_radius);
    	return var_22;
    }
    var_23 = df::select(var_15, var_11, var_22);
    var_24 = df::index(var_p, var_5);
    var_25 = df::index(var_p, var_7);
    var_26 = df::float3(var_13, var_24, var_25);
    var_27 = df::length(var_26);
    var_28 = df::sub(var_27, var_radius);
    return var_28;
    

}

void adj_capsule_sdf_cpu_func(
	float var_radius,
	float var_half_width,
	df::float3 var_p, 
	float & adj_radius,
	float & adj_half_width,
	df::float3 & adj_p,
	float & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    bool var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    const int var_7 = 2;
    float var_8;
    df::float3 var_9;
    float var_10;
    float var_11;
    float var_12;
    const float var_13 = 0.0;
    float var_14;
    bool var_15;
    float var_16;
    float var_17;
    float var_18;
    float var_19;
    df::float3 var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    df::float3 var_26;
    float var_27;
    float var_28;
    //---------
    // dual vars
    int adj_0 = 0;
    float adj_1 = 0;
    bool adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    int adj_5 = 0;
    float adj_6 = 0;
    int adj_7 = 0;
    float adj_8 = 0;
    df::float3 adj_9 = 0;
    float adj_10 = 0;
    float adj_11 = 0;
    float adj_12 = 0;
    float adj_13 = 0;
    float adj_14 = 0;
    bool adj_15 = 0;
    float adj_16 = 0;
    float adj_17 = 0;
    float adj_18 = 0;
    float adj_19 = 0;
    df::float3 adj_20 = 0;
    float adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    df::float3 adj_26 = 0;
    float adj_27 = 0;
    float adj_28 = 0;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = (var_1 > var_half_width);
    if (var_2) {
    	var_3 = df::index(var_p, var_0);
    	var_4 = df::sub(var_3, var_half_width);
    	var_6 = df::index(var_p, var_5);
    	var_8 = df::index(var_p, var_7);
    	var_9 = df::float3(var_4, var_6, var_8);
    	var_10 = df::length(var_9);
    	var_11 = df::sub(var_10, var_radius);
    	goto label0;
    }
    var_12 = df::index(var_p, var_0);
    var_14 = df::sub(var_13, var_half_width);
    var_15 = (var_12 < var_14);
    if (var_15) {
    	var_16 = df::index(var_p, var_0);
    	var_17 = df::add(var_16, var_half_width);
    	var_18 = df::index(var_p, var_5);
    	var_19 = df::index(var_p, var_7);
    	var_20 = df::float3(var_17, var_18, var_19);
    	var_21 = df::length(var_20);
    	var_22 = df::sub(var_21, var_radius);
    	goto label1;
    }
    var_23 = df::select(var_15, var_11, var_22);
    var_24 = df::index(var_p, var_5);
    var_25 = df::index(var_p, var_7);
    var_26 = df::float3(var_13, var_24, var_25);
    var_27 = df::length(var_26);
    var_28 = df::sub(var_27, var_radius);
    goto label2;
    //---------
    // reverse
    label2:;
    adj_28 += adj_ret;
    df::adj_sub(var_27, var_radius, adj_27, adj_radius, adj_28);
    df::adj_length(var_26, adj_26, adj_27);
    df::adj_float3(var_13, var_24, var_25, adj_13, adj_24, adj_25, adj_26);
    df::adj_index(var_p, var_7, adj_p, adj_7, adj_25);
    df::adj_index(var_p, var_5, adj_p, adj_5, adj_24);
    df::adj_select(var_15, var_11, var_22, adj_15, adj_11, adj_22, adj_23);
    if (var_15) {
    	label1:;
    	adj_22 += adj_ret;
    	df::adj_sub(var_21, var_radius, adj_21, adj_radius, adj_22);
    	df::adj_length(var_20, adj_20, adj_21);
    	df::adj_float3(var_17, var_18, var_19, adj_17, adj_18, adj_19, adj_20);
    	df::adj_index(var_p, var_7, adj_p, adj_7, adj_19);
    	df::adj_index(var_p, var_5, adj_p, adj_5, adj_18);
    	df::adj_add(var_16, var_half_width, adj_16, adj_half_width, adj_17);
    	df::adj_index(var_p, var_0, adj_p, adj_0, adj_16);
    }
    df::adj_sub(var_13, var_half_width, adj_13, adj_half_width, adj_14);
    df::adj_index(var_p, var_0, adj_p, adj_0, adj_12);
    if (var_2) {
    	label0:;
    	adj_11 += adj_ret;
    	df::adj_sub(var_10, var_radius, adj_10, adj_radius, adj_11);
    	df::adj_length(var_9, adj_9, adj_10);
    	df::adj_float3(var_4, var_6, var_8, adj_4, adj_6, adj_8, adj_9);
    	df::adj_index(var_p, var_7, adj_p, adj_7, adj_8);
    	df::adj_index(var_p, var_5, adj_p, adj_5, adj_6);
    	df::adj_sub(var_3, var_half_width, adj_3, adj_half_width, adj_4);
    	df::adj_index(var_p, var_0, adj_p, adj_0, adj_3);
    }
    df::adj_index(var_p, var_0, adj_p, adj_0, adj_1);
    return;

}


df::float3 capsule_sdf_grad_cpu_func(
	float var_radius,
	float var_half_width,
	df::float3 var_p)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    bool var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    const int var_7 = 2;
    float var_8;
    df::float3 var_9;
    df::float3 var_10;
    float var_11;
    const float var_12 = 0.0;
    float var_13;
    bool var_14;
    float var_15;
    float var_16;
    float var_17;
    float var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    float var_22;
    float var_23;
    df::float3 var_24;
    df::float3 var_25;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = (var_1 > var_half_width);
    if (var_2) {
    	var_3 = df::index(var_p, var_0);
    	var_4 = df::sub(var_3, var_half_width);
    	var_6 = df::index(var_p, var_5);
    	var_8 = df::index(var_p, var_7);
    	var_9 = df::float3(var_4, var_6, var_8);
    	var_10 = df::normalize(var_9);
    	return var_10;
    }
    var_11 = df::index(var_p, var_0);
    var_13 = df::sub(var_12, var_half_width);
    var_14 = (var_11 < var_13);
    if (var_14) {
    	var_15 = df::index(var_p, var_0);
    	var_16 = df::add(var_15, var_half_width);
    	var_17 = df::index(var_p, var_5);
    	var_18 = df::index(var_p, var_7);
    	var_19 = df::float3(var_16, var_17, var_18);
    	var_20 = df::normalize(var_19);
    	return var_20;
    }
    var_21 = df::select(var_14, var_10, var_20);
    var_22 = df::index(var_p, var_5);
    var_23 = df::index(var_p, var_7);
    var_24 = df::float3(var_12, var_22, var_23);
    var_25 = df::normalize(var_24);
    return var_25;
    

}

void adj_capsule_sdf_grad_cpu_func(
	float var_radius,
	float var_half_width,
	df::float3 var_p, 
	float & adj_radius,
	float & adj_half_width,
	df::float3 & adj_p,
	df::float3 & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    float var_1;
    bool var_2;
    float var_3;
    float var_4;
    const int var_5 = 1;
    float var_6;
    const int var_7 = 2;
    float var_8;
    df::float3 var_9;
    df::float3 var_10;
    float var_11;
    const float var_12 = 0.0;
    float var_13;
    bool var_14;
    float var_15;
    float var_16;
    float var_17;
    float var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    float var_22;
    float var_23;
    df::float3 var_24;
    df::float3 var_25;
    //---------
    // dual vars
    int adj_0 = 0;
    float adj_1 = 0;
    bool adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    int adj_5 = 0;
    float adj_6 = 0;
    int adj_7 = 0;
    float adj_8 = 0;
    df::float3 adj_9 = 0;
    df::float3 adj_10 = 0;
    float adj_11 = 0;
    float adj_12 = 0;
    float adj_13 = 0;
    bool adj_14 = 0;
    float adj_15 = 0;
    float adj_16 = 0;
    float adj_17 = 0;
    float adj_18 = 0;
    df::float3 adj_19 = 0;
    df::float3 adj_20 = 0;
    df::float3 adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    df::float3 adj_24 = 0;
    df::float3 adj_25 = 0;
    //---------
    // forward
    var_1 = df::index(var_p, var_0);
    var_2 = (var_1 > var_half_width);
    if (var_2) {
    	var_3 = df::index(var_p, var_0);
    	var_4 = df::sub(var_3, var_half_width);
    	var_6 = df::index(var_p, var_5);
    	var_8 = df::index(var_p, var_7);
    	var_9 = df::float3(var_4, var_6, var_8);
    	var_10 = df::normalize(var_9);
    	goto label0;
    }
    var_11 = df::index(var_p, var_0);
    var_13 = df::sub(var_12, var_half_width);
    var_14 = (var_11 < var_13);
    if (var_14) {
    	var_15 = df::index(var_p, var_0);
    	var_16 = df::add(var_15, var_half_width);
    	var_17 = df::index(var_p, var_5);
    	var_18 = df::index(var_p, var_7);
    	var_19 = df::float3(var_16, var_17, var_18);
    	var_20 = df::normalize(var_19);
    	goto label1;
    }
    var_21 = df::select(var_14, var_10, var_20);
    var_22 = df::index(var_p, var_5);
    var_23 = df::index(var_p, var_7);
    var_24 = df::float3(var_12, var_22, var_23);
    var_25 = df::normalize(var_24);
    goto label2;
    //---------
    // reverse
    label2:;
    adj_25 += adj_ret;
    df::adj_normalize(var_24, adj_24, adj_25);
    df::adj_float3(var_12, var_22, var_23, adj_12, adj_22, adj_23, adj_24);
    df::adj_index(var_p, var_7, adj_p, adj_7, adj_23);
    df::adj_index(var_p, var_5, adj_p, adj_5, adj_22);
    df::adj_select(var_14, var_10, var_20, adj_14, adj_10, adj_20, adj_21);
    if (var_14) {
    	label1:;
    	adj_20 += adj_ret;
    	df::adj_normalize(var_19, adj_19, adj_20);
    	df::adj_float3(var_16, var_17, var_18, adj_16, adj_17, adj_18, adj_19);
    	df::adj_index(var_p, var_7, adj_p, adj_7, adj_18);
    	df::adj_index(var_p, var_5, adj_p, adj_5, adj_17);
    	df::adj_add(var_15, var_half_width, adj_15, adj_half_width, adj_16);
    	df::adj_index(var_p, var_0, adj_p, adj_0, adj_15);
    }
    df::adj_sub(var_12, var_half_width, adj_12, adj_half_width, adj_13);
    df::adj_index(var_p, var_0, adj_p, adj_0, adj_11);
    if (var_2) {
    	label0:;
    	adj_10 += adj_ret;
    	df::adj_normalize(var_9, adj_9, adj_10);
    	df::adj_float3(var_4, var_6, var_8, adj_4, adj_6, adj_8, adj_9);
    	df::adj_index(var_p, var_7, adj_p, adj_7, adj_8);
    	df::adj_index(var_p, var_5, adj_p, adj_5, adj_6);
    	df::adj_sub(var_3, var_half_width, adj_3, adj_half_width, adj_4);
    	df::adj_index(var_p, var_0, adj_p, adj_0, adj_3);
    }
    df::adj_index(var_p, var_0, adj_p, adj_0, adj_1);
    return;

}


spatial_vector spatial_transform_twist_cpu_func(
	spatial_transform var_t,
	spatial_vector var_x)
{
        //---------
    // primal vars
    quat var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    df::float3 var_7;
    spatial_vector var_8;
    //---------
    // forward
    var_0 = df::spatial_transform_get_rotation(var_t);
    var_1 = df::spatial_transform_get_translation(var_t);
    var_2 = df::spatial_top(var_x);
    var_3 = df::spatial_bottom(var_x);
    var_4 = df::rotate(var_0, var_2);
    var_5 = df::rotate(var_0, var_3);
    var_6 = df::cross(var_1, var_4);
    var_7 = df::add(var_5, var_6);
    var_8 = df::spatial_vector(var_4, var_7);
    return var_8;
    

}

void adj_spatial_transform_twist_cpu_func(
	spatial_transform var_t,
	spatial_vector var_x, 
	spatial_transform & adj_t,
	spatial_vector & adj_x,
	spatial_vector & adj_ret)
{
        //---------
    // primal vars
    quat var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    df::float3 var_7;
    spatial_vector var_8;
    //---------
    // dual vars
    quat adj_0 = 0;
    df::float3 adj_1 = 0;
    df::float3 adj_2 = 0;
    df::float3 adj_3 = 0;
    df::float3 adj_4 = 0;
    df::float3 adj_5 = 0;
    df::float3 adj_6 = 0;
    df::float3 adj_7 = 0;
    spatial_vector adj_8 = 0;
    //---------
    // forward
    var_0 = df::spatial_transform_get_rotation(var_t);
    var_1 = df::spatial_transform_get_translation(var_t);
    var_2 = df::spatial_top(var_x);
    var_3 = df::spatial_bottom(var_x);
    var_4 = df::rotate(var_0, var_2);
    var_5 = df::rotate(var_0, var_3);
    var_6 = df::cross(var_1, var_4);
    var_7 = df::add(var_5, var_6);
    var_8 = df::spatial_vector(var_4, var_7);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_8 += adj_ret;
    df::adj_spatial_vector(var_4, var_7, adj_4, adj_7, adj_8);
    df::adj_add(var_5, var_6, adj_5, adj_6, adj_7);
    df::adj_cross(var_1, var_4, adj_1, adj_4, adj_6);
    df::adj_rotate(var_0, var_3, adj_0, adj_3, adj_5);
    df::adj_rotate(var_0, var_2, adj_0, adj_2, adj_4);
    df::adj_spatial_bottom(var_x, adj_x, adj_3);
    df::adj_spatial_top(var_x, adj_x, adj_2);
    df::adj_spatial_transform_get_translation(var_t, adj_t, adj_1);
    df::adj_spatial_transform_get_rotation(var_t, adj_t, adj_0);
    return;

}


spatial_vector spatial_transform_wrench_cpu_func(
	spatial_transform var_t,
	spatial_vector var_x)
{
        //---------
    // primal vars
    quat var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    df::float3 var_7;
    spatial_vector var_8;
    //---------
    // forward
    var_0 = df::spatial_transform_get_rotation(var_t);
    var_1 = df::spatial_transform_get_translation(var_t);
    var_2 = df::spatial_top(var_x);
    var_3 = df::spatial_bottom(var_x);
    var_4 = df::rotate(var_0, var_3);
    var_5 = df::rotate(var_0, var_2);
    var_6 = df::cross(var_1, var_4);
    var_7 = df::add(var_5, var_6);
    var_8 = df::spatial_vector(var_7, var_4);
    return var_8;
    

}

void adj_spatial_transform_wrench_cpu_func(
	spatial_transform var_t,
	spatial_vector var_x, 
	spatial_transform & adj_t,
	spatial_vector & adj_x,
	spatial_vector & adj_ret)
{
        //---------
    // primal vars
    quat var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    df::float3 var_7;
    spatial_vector var_8;
    //---------
    // dual vars
    quat adj_0 = 0;
    df::float3 adj_1 = 0;
    df::float3 adj_2 = 0;
    df::float3 adj_3 = 0;
    df::float3 adj_4 = 0;
    df::float3 adj_5 = 0;
    df::float3 adj_6 = 0;
    df::float3 adj_7 = 0;
    spatial_vector adj_8 = 0;
    //---------
    // forward
    var_0 = df::spatial_transform_get_rotation(var_t);
    var_1 = df::spatial_transform_get_translation(var_t);
    var_2 = df::spatial_top(var_x);
    var_3 = df::spatial_bottom(var_x);
    var_4 = df::rotate(var_0, var_3);
    var_5 = df::rotate(var_0, var_2);
    var_6 = df::cross(var_1, var_4);
    var_7 = df::add(var_5, var_6);
    var_8 = df::spatial_vector(var_7, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_8 += adj_ret;
    df::adj_spatial_vector(var_7, var_4, adj_7, adj_4, adj_8);
    df::adj_add(var_5, var_6, adj_5, adj_6, adj_7);
    df::adj_cross(var_1, var_4, adj_1, adj_4, adj_6);
    df::adj_rotate(var_0, var_2, adj_0, adj_2, adj_5);
    df::adj_rotate(var_0, var_3, adj_0, adj_3, adj_4);
    df::adj_spatial_bottom(var_x, adj_x, adj_3);
    df::adj_spatial_top(var_x, adj_x, adj_2);
    df::adj_spatial_transform_get_translation(var_t, adj_t, adj_1);
    df::adj_spatial_transform_get_rotation(var_t, adj_t, adj_0);
    return;

}


spatial_transform spatial_transform_inverse_cpu_func(
	spatial_transform var_t)
{
        //---------
    // primal vars
    df::float3 var_0;
    quat var_1;
    quat var_2;
    df::float3 var_3;
    const float var_4 = 0.0;
    const float var_5 = 1.0;
    float var_6;
    df::float3 var_7;
    spatial_transform var_8;
    //---------
    // forward
    var_0 = df::spatial_transform_get_translation(var_t);
    var_1 = df::spatial_transform_get_rotation(var_t);
    var_2 = df::inverse(var_1);
    var_3 = df::rotate(var_2, var_0);
    var_6 = df::sub(var_4, var_5);
    var_7 = df::mul(var_3, var_6);
    var_8 = df::spatial_transform(var_7, var_2);
    return var_8;
    

}

void adj_spatial_transform_inverse_cpu_func(
	spatial_transform var_t, 
	spatial_transform & adj_t,
	spatial_transform & adj_ret)
{
        //---------
    // primal vars
    df::float3 var_0;
    quat var_1;
    quat var_2;
    df::float3 var_3;
    const float var_4 = 0.0;
    const float var_5 = 1.0;
    float var_6;
    df::float3 var_7;
    spatial_transform var_8;
    //---------
    // dual vars
    df::float3 adj_0 = 0;
    quat adj_1 = 0;
    quat adj_2 = 0;
    df::float3 adj_3 = 0;
    float adj_4 = 0;
    float adj_5 = 0;
    float adj_6 = 0;
    df::float3 adj_7 = 0;
    spatial_transform adj_8 = 0;
    //---------
    // forward
    var_0 = df::spatial_transform_get_translation(var_t);
    var_1 = df::spatial_transform_get_rotation(var_t);
    var_2 = df::inverse(var_1);
    var_3 = df::rotate(var_2, var_0);
    var_6 = df::sub(var_4, var_5);
    var_7 = df::mul(var_3, var_6);
    var_8 = df::spatial_transform(var_7, var_2);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_8 += adj_ret;
    df::adj_spatial_transform(var_7, var_2, adj_7, adj_2, adj_8);
    df::adj_mul(var_3, var_6, adj_3, adj_6, adj_7);
    df::adj_sub(var_4, var_5, adj_4, adj_5, adj_6);
    df::adj_rotate(var_2, var_0, adj_2, adj_0, adj_3);
    df::adj_inverse(var_1, adj_1, adj_2);
    df::adj_spatial_transform_get_rotation(var_t, adj_t, adj_1);
    df::adj_spatial_transform_get_translation(var_t, adj_t, adj_0);
    return;

}


spatial_matrix spatial_transform_inertia_cpu_func(
	spatial_transform var_t,
	spatial_matrix var_I)
{
        //---------
    // primal vars
    spatial_transform var_0;
    quat var_1;
    df::float3 var_2;
    const float var_3 = 1.0;
    const float var_4 = 0.0;
    df::float3 var_5;
    df::float3 var_6;
    df::float3 var_7;
    df::float3 var_8;
    df::float3 var_9;
    df::float3 var_10;
    mat33 var_11;
    mat33 var_12;
    mat33 var_13;
    spatial_matrix var_14;
    spatial_matrix var_15;
    spatial_matrix var_16;
    spatial_matrix var_17;
    //---------
    // forward
    var_0 = spatial_transform_inverse_cpu_func(var_t);
    var_1 = df::spatial_transform_get_rotation(var_0);
    var_2 = df::spatial_transform_get_translation(var_0);
    var_5 = df::float3(var_3, var_4, var_4);
    var_6 = df::rotate(var_1, var_5);
    var_7 = df::float3(var_4, var_3, var_4);
    var_8 = df::rotate(var_1, var_7);
    var_9 = df::float3(var_4, var_4, var_3);
    var_10 = df::rotate(var_1, var_9);
    var_11 = df::mat33(var_6, var_8, var_10);
    var_12 = df::skew(var_2);
    var_13 = df::mul(var_12, var_11);
    var_14 = df::spatial_adjoint(var_11, var_13);
    var_15 = df::transpose(var_14);
    var_16 = df::mul(var_15, var_I);
    var_17 = df::mul(var_16, var_14);
    return var_17;
    

}

void adj_spatial_transform_inertia_cpu_func(
	spatial_transform var_t,
	spatial_matrix var_I, 
	spatial_transform & adj_t,
	spatial_matrix & adj_I,
	spatial_matrix & adj_ret)
{
        //---------
    // primal vars
    spatial_transform var_0;
    quat var_1;
    df::float3 var_2;
    const float var_3 = 1.0;
    const float var_4 = 0.0;
    df::float3 var_5;
    df::float3 var_6;
    df::float3 var_7;
    df::float3 var_8;
    df::float3 var_9;
    df::float3 var_10;
    mat33 var_11;
    mat33 var_12;
    mat33 var_13;
    spatial_matrix var_14;
    spatial_matrix var_15;
    spatial_matrix var_16;
    spatial_matrix var_17;
    //---------
    // dual vars
    spatial_transform adj_0 = 0;
    quat adj_1 = 0;
    df::float3 adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    df::float3 adj_5 = 0;
    df::float3 adj_6 = 0;
    df::float3 adj_7 = 0;
    df::float3 adj_8 = 0;
    df::float3 adj_9 = 0;
    df::float3 adj_10 = 0;
    mat33 adj_11 = 0;
    mat33 adj_12 = 0;
    mat33 adj_13 = 0;
    spatial_matrix adj_14 = 0;
    spatial_matrix adj_15 = 0;
    spatial_matrix adj_16 = 0;
    spatial_matrix adj_17 = 0;
    //---------
    // forward
    var_0 = spatial_transform_inverse_cpu_func(var_t);
    var_1 = df::spatial_transform_get_rotation(var_0);
    var_2 = df::spatial_transform_get_translation(var_0);
    var_5 = df::float3(var_3, var_4, var_4);
    var_6 = df::rotate(var_1, var_5);
    var_7 = df::float3(var_4, var_3, var_4);
    var_8 = df::rotate(var_1, var_7);
    var_9 = df::float3(var_4, var_4, var_3);
    var_10 = df::rotate(var_1, var_9);
    var_11 = df::mat33(var_6, var_8, var_10);
    var_12 = df::skew(var_2);
    var_13 = df::mul(var_12, var_11);
    var_14 = df::spatial_adjoint(var_11, var_13);
    var_15 = df::transpose(var_14);
    var_16 = df::mul(var_15, var_I);
    var_17 = df::mul(var_16, var_14);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_17 += adj_ret;
    df::adj_mul(var_16, var_14, adj_16, adj_14, adj_17);
    df::adj_mul(var_15, var_I, adj_15, adj_I, adj_16);
    df::adj_transpose(var_14, adj_14, adj_15);
    df::adj_spatial_adjoint(var_11, var_13, adj_11, adj_13, adj_14);
    df::adj_mul(var_12, var_11, adj_12, adj_11, adj_13);
    df::adj_skew(var_2, adj_2, adj_12);
    df::adj_mat33(var_6, var_8, var_10, adj_6, adj_8, adj_10, adj_11);
    df::adj_rotate(var_1, var_9, adj_1, adj_9, adj_10);
    df::adj_float3(var_4, var_4, var_3, adj_4, adj_4, adj_3, adj_9);
    df::adj_rotate(var_1, var_7, adj_1, adj_7, adj_8);
    df::adj_float3(var_4, var_3, var_4, adj_4, adj_3, adj_4, adj_7);
    df::adj_rotate(var_1, var_5, adj_1, adj_5, adj_6);
    df::adj_float3(var_3, var_4, var_4, adj_3, adj_4, adj_4, adj_5);
    df::adj_spatial_transform_get_translation(var_0, adj_0, adj_2);
    df::adj_spatial_transform_get_rotation(var_0, adj_0, adj_1);
    adj_spatial_transform_inverse_cpu_func(var_t, adj_t, adj_0);
    return;

}


int compute_muscle_force_cpu_func(
	int var_i,
	spatial_transform* var_body_X_s,
	spatial_vector* var_body_v_s,
	int* var_muscle_links,
	df::float3* var_muscle_points,
	float var_muscle_activation,
	spatial_vector* var_body_f_s)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 1;
    int var_2;
    int var_3;
    bool var_4;
    const int var_5 = 0;
    df::float3 var_6;
    int var_7;
    df::float3 var_8;
    spatial_transform var_9;
    spatial_transform var_10;
    df::float3 var_11;
    df::float3 var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    spatial_vector var_17;
    df::float3 var_18;
    spatial_vector var_19;
    //---------
    // forward
    var_0 = df::load(var_muscle_links, var_i);
    var_2 = df::add(var_i, var_1);
    var_3 = df::load(var_muscle_links, var_2);
    var_4 = (var_0 == var_3);
    if (var_4) {
    	return var_5;
    }
    var_6 = df::load(var_muscle_points, var_i);
    var_7 = df::add(var_i, var_1);
    var_8 = df::load(var_muscle_points, var_7);
    var_9 = df::load(var_body_X_s, var_0);
    var_10 = df::load(var_body_X_s, var_3);
    var_11 = df::spatial_transform_point(var_9, var_6);
    var_12 = df::spatial_transform_point(var_10, var_8);
    var_13 = df::sub(var_12, var_11);
    var_14 = df::normalize(var_13);
    var_15 = df::mul(var_14, var_muscle_activation);
    var_16 = df::cross(var_11, var_15);
    var_17 = df::spatial_vector(var_16, var_15);
    df::atomic_sub(var_body_f_s, var_0, var_17);
    var_18 = df::cross(var_12, var_15);
    var_19 = df::spatial_vector(var_18, var_15);
    df::atomic_add(var_body_f_s, var_3, var_19);
    return var_5;
    

}

void adj_compute_muscle_force_cpu_func(
	int var_i,
	spatial_transform* var_body_X_s,
	spatial_vector* var_body_v_s,
	int* var_muscle_links,
	df::float3* var_muscle_points,
	float var_muscle_activation,
	spatial_vector* var_body_f_s, 
	int & adj_i,
	spatial_transform* adj_body_X_s,
	spatial_vector* adj_body_v_s,
	int* adj_muscle_links,
	df::float3* adj_muscle_points,
	float & adj_muscle_activation,
	spatial_vector* adj_body_f_s,
	int & adj_ret)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 1;
    int var_2;
    int var_3;
    bool var_4;
    const int var_5 = 0;
    df::float3 var_6;
    int var_7;
    df::float3 var_8;
    spatial_transform var_9;
    spatial_transform var_10;
    df::float3 var_11;
    df::float3 var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    spatial_vector var_17;
    df::float3 var_18;
    spatial_vector var_19;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    bool adj_4 = 0;
    int adj_5 = 0;
    df::float3 adj_6 = 0;
    int adj_7 = 0;
    df::float3 adj_8 = 0;
    spatial_transform adj_9 = 0;
    spatial_transform adj_10 = 0;
    df::float3 adj_11 = 0;
    df::float3 adj_12 = 0;
    df::float3 adj_13 = 0;
    df::float3 adj_14 = 0;
    df::float3 adj_15 = 0;
    df::float3 adj_16 = 0;
    spatial_vector adj_17 = 0;
    df::float3 adj_18 = 0;
    spatial_vector adj_19 = 0;
    //---------
    // forward
    var_0 = df::load(var_muscle_links, var_i);
    var_2 = df::add(var_i, var_1);
    var_3 = df::load(var_muscle_links, var_2);
    var_4 = (var_0 == var_3);
    if (var_4) {
    	goto label0;
    }
    var_6 = df::load(var_muscle_points, var_i);
    var_7 = df::add(var_i, var_1);
    var_8 = df::load(var_muscle_points, var_7);
    var_9 = df::load(var_body_X_s, var_0);
    var_10 = df::load(var_body_X_s, var_3);
    var_11 = df::spatial_transform_point(var_9, var_6);
    var_12 = df::spatial_transform_point(var_10, var_8);
    var_13 = df::sub(var_12, var_11);
    var_14 = df::normalize(var_13);
    var_15 = df::mul(var_14, var_muscle_activation);
    var_16 = df::cross(var_11, var_15);
    var_17 = df::spatial_vector(var_16, var_15);
    df::atomic_sub(var_body_f_s, var_0, var_17);
    var_18 = df::cross(var_12, var_15);
    var_19 = df::spatial_vector(var_18, var_15);
    df::atomic_add(var_body_f_s, var_3, var_19);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_5 += adj_ret;
    df::adj_atomic_add(var_body_f_s, var_3, var_19, adj_body_f_s, adj_3, adj_19);
    df::adj_spatial_vector(var_18, var_15, adj_18, adj_15, adj_19);
    df::adj_cross(var_12, var_15, adj_12, adj_15, adj_18);
    df::adj_atomic_sub(var_body_f_s, var_0, var_17, adj_body_f_s, adj_0, adj_17);
    df::adj_spatial_vector(var_16, var_15, adj_16, adj_15, adj_17);
    df::adj_cross(var_11, var_15, adj_11, adj_15, adj_16);
    df::adj_mul(var_14, var_muscle_activation, adj_14, adj_muscle_activation, adj_15);
    df::adj_normalize(var_13, adj_13, adj_14);
    df::adj_sub(var_12, var_11, adj_12, adj_11, adj_13);
    df::adj_spatial_transform_point(var_10, var_8, adj_10, adj_8, adj_12);
    df::adj_spatial_transform_point(var_9, var_6, adj_9, adj_6, adj_11);
    df::adj_load(var_body_X_s, var_3, adj_body_X_s, adj_3, adj_10);
    df::adj_load(var_body_X_s, var_0, adj_body_X_s, adj_0, adj_9);
    df::adj_load(var_muscle_points, var_7, adj_muscle_points, adj_7, adj_8);
    df::adj_add(var_i, var_1, adj_i, adj_1, adj_7);
    df::adj_load(var_muscle_points, var_i, adj_muscle_points, adj_i, adj_6);
    if (var_4) {
    	label0:;
    	adj_5 += adj_ret;
    }
    df::adj_load(var_muscle_links, var_2, adj_muscle_links, adj_2, adj_3);
    df::adj_add(var_i, var_1, adj_i, adj_1, adj_2);
    df::adj_load(var_muscle_links, var_i, adj_muscle_links, adj_i, adj_0);
    return;

}


spatial_transform jcalc_transform_cpu_func(
	int var_type,
	df::float3 var_axis,
	float* var_joint_q,
	int var_start)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    float var_2;
    df::float3 var_3;
    quat var_4;
    spatial_transform var_5;
    const int var_6 = 1;
    bool var_7;
    float var_8;
    const float var_9 = 0.0;
    df::float3 var_10;
    quat var_11;
    spatial_transform var_12;
    float var_13;
    spatial_transform var_14;
    spatial_transform var_15;
    const int var_16 = 2;
    bool var_17;
    int var_18;
    float var_19;
    int var_20;
    float var_21;
    int var_22;
    float var_23;
    const int var_24 = 3;
    int var_25;
    float var_26;
    df::float3 var_27;
    quat var_28;
    spatial_transform var_29;
    spatial_transform var_30;
    spatial_transform var_31;
    bool var_32;
    spatial_transform var_33;
    spatial_transform var_34;
    spatial_transform var_35;
    const int var_36 = 4;
    bool var_37;
    int var_38;
    float var_39;
    int var_40;
    float var_41;
    int var_42;
    float var_43;
    int var_44;
    float var_45;
    int var_46;
    float var_47;
    const int var_48 = 5;
    int var_49;
    float var_50;
    const int var_51 = 6;
    int var_52;
    float var_53;
    df::float3 var_54;
    quat var_55;
    spatial_transform var_56;
    spatial_transform var_57;
    spatial_transform var_58;
    float var_59;
    float var_60;
    float var_61;
    float var_62;
    spatial_transform var_63;
    //---------
    // forward
    var_1 = (var_type == var_0);
    if (var_1) {
    	var_2 = df::load(var_joint_q, var_start);
    	var_3 = df::mul(var_axis, var_2);
    	var_4 = df::quat_identity();
    	var_5 = df::spatial_transform(var_3, var_4);
    	return var_5;
    }
    var_7 = (var_type == var_6);
    if (var_7) {
    	var_8 = df::load(var_joint_q, var_start);
    	var_10 = df::float3(var_9, var_9, var_9);
    	var_11 = df::quat_from_axis_angle(var_axis, var_8);
    	var_12 = df::spatial_transform(var_10, var_11);
    	return var_12;
    }
    var_13 = df::select(var_7, var_2, var_8);
    var_14 = df::select(var_7, var_5, var_12);
    var_15 = df::select(var_7, var_5, var_12);
    var_17 = (var_type == var_16);
    if (var_17) {
    	var_18 = df::add(var_start, var_0);
    	var_19 = df::load(var_joint_q, var_18);
    	var_20 = df::add(var_start, var_6);
    	var_21 = df::load(var_joint_q, var_20);
    	var_22 = df::add(var_start, var_16);
    	var_23 = df::load(var_joint_q, var_22);
    	var_25 = df::add(var_start, var_24);
    	var_26 = df::load(var_joint_q, var_25);
    	var_27 = df::float3(var_9, var_9, var_9);
    	var_28 = df::quat(var_19, var_21, var_23, var_26);
    	var_29 = df::spatial_transform(var_27, var_28);
    	return var_29;
    }
    var_30 = df::select(var_17, var_14, var_29);
    var_31 = df::select(var_17, var_15, var_29);
    var_32 = (var_type == var_24);
    if (var_32) {
    	var_33 = df::spatial_transform_identity();
    	return var_33;
    }
    var_34 = df::select(var_32, var_30, var_33);
    var_35 = df::select(var_32, var_31, var_33);
    var_37 = (var_type == var_36);
    if (var_37) {
    	var_38 = df::add(var_start, var_0);
    	var_39 = df::load(var_joint_q, var_38);
    	var_40 = df::add(var_start, var_6);
    	var_41 = df::load(var_joint_q, var_40);
    	var_42 = df::add(var_start, var_16);
    	var_43 = df::load(var_joint_q, var_42);
    	var_44 = df::add(var_start, var_24);
    	var_45 = df::load(var_joint_q, var_44);
    	var_46 = df::add(var_start, var_36);
    	var_47 = df::load(var_joint_q, var_46);
    	var_49 = df::add(var_start, var_48);
    	var_50 = df::load(var_joint_q, var_49);
    	var_52 = df::add(var_start, var_51);
    	var_53 = df::load(var_joint_q, var_52);
    	var_54 = df::float3(var_39, var_41, var_43);
    	var_55 = df::quat(var_45, var_47, var_50, var_53);
    	var_56 = df::spatial_transform(var_54, var_55);
    	return var_56;
    }
    var_57 = df::select(var_37, var_34, var_56);
    var_58 = df::select(var_37, var_35, var_56);
    var_59 = df::select(var_37, var_19, var_45);
    var_60 = df::select(var_37, var_21, var_47);
    var_61 = df::select(var_37, var_23, var_50);
    var_62 = df::select(var_37, var_26, var_53);
    var_63 = df::spatial_transform_identity();
    return var_63;
    

}

void adj_jcalc_transform_cpu_func(
	int var_type,
	df::float3 var_axis,
	float* var_joint_q,
	int var_start, 
	int & adj_type,
	df::float3 & adj_axis,
	float* adj_joint_q,
	int & adj_start,
	spatial_transform & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    float var_2;
    df::float3 var_3;
    quat var_4;
    spatial_transform var_5;
    const int var_6 = 1;
    bool var_7;
    float var_8;
    const float var_9 = 0.0;
    df::float3 var_10;
    quat var_11;
    spatial_transform var_12;
    float var_13;
    spatial_transform var_14;
    spatial_transform var_15;
    const int var_16 = 2;
    bool var_17;
    int var_18;
    float var_19;
    int var_20;
    float var_21;
    int var_22;
    float var_23;
    const int var_24 = 3;
    int var_25;
    float var_26;
    df::float3 var_27;
    quat var_28;
    spatial_transform var_29;
    spatial_transform var_30;
    spatial_transform var_31;
    bool var_32;
    spatial_transform var_33;
    spatial_transform var_34;
    spatial_transform var_35;
    const int var_36 = 4;
    bool var_37;
    int var_38;
    float var_39;
    int var_40;
    float var_41;
    int var_42;
    float var_43;
    int var_44;
    float var_45;
    int var_46;
    float var_47;
    const int var_48 = 5;
    int var_49;
    float var_50;
    const int var_51 = 6;
    int var_52;
    float var_53;
    df::float3 var_54;
    quat var_55;
    spatial_transform var_56;
    spatial_transform var_57;
    spatial_transform var_58;
    float var_59;
    float var_60;
    float var_61;
    float var_62;
    spatial_transform var_63;
    //---------
    // dual vars
    int adj_0 = 0;
    bool adj_1 = 0;
    float adj_2 = 0;
    df::float3 adj_3 = 0;
    quat adj_4 = 0;
    spatial_transform adj_5 = 0;
    int adj_6 = 0;
    bool adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    df::float3 adj_10 = 0;
    quat adj_11 = 0;
    spatial_transform adj_12 = 0;
    float adj_13 = 0;
    spatial_transform adj_14 = 0;
    spatial_transform adj_15 = 0;
    int adj_16 = 0;
    bool adj_17 = 0;
    int adj_18 = 0;
    float adj_19 = 0;
    int adj_20 = 0;
    float adj_21 = 0;
    int adj_22 = 0;
    float adj_23 = 0;
    int adj_24 = 0;
    int adj_25 = 0;
    float adj_26 = 0;
    df::float3 adj_27 = 0;
    quat adj_28 = 0;
    spatial_transform adj_29 = 0;
    spatial_transform adj_30 = 0;
    spatial_transform adj_31 = 0;
    bool adj_32 = 0;
    spatial_transform adj_33 = 0;
    spatial_transform adj_34 = 0;
    spatial_transform adj_35 = 0;
    int adj_36 = 0;
    bool adj_37 = 0;
    int adj_38 = 0;
    float adj_39 = 0;
    int adj_40 = 0;
    float adj_41 = 0;
    int adj_42 = 0;
    float adj_43 = 0;
    int adj_44 = 0;
    float adj_45 = 0;
    int adj_46 = 0;
    float adj_47 = 0;
    int adj_48 = 0;
    int adj_49 = 0;
    float adj_50 = 0;
    int adj_51 = 0;
    int adj_52 = 0;
    float adj_53 = 0;
    df::float3 adj_54 = 0;
    quat adj_55 = 0;
    spatial_transform adj_56 = 0;
    spatial_transform adj_57 = 0;
    spatial_transform adj_58 = 0;
    float adj_59 = 0;
    float adj_60 = 0;
    float adj_61 = 0;
    float adj_62 = 0;
    spatial_transform adj_63 = 0;
    //---------
    // forward
    var_1 = (var_type == var_0);
    if (var_1) {
    	var_2 = df::load(var_joint_q, var_start);
    	var_3 = df::mul(var_axis, var_2);
    	var_4 = df::quat_identity();
    	var_5 = df::spatial_transform(var_3, var_4);
    	goto label0;
    }
    var_7 = (var_type == var_6);
    if (var_7) {
    	var_8 = df::load(var_joint_q, var_start);
    	var_10 = df::float3(var_9, var_9, var_9);
    	var_11 = df::quat_from_axis_angle(var_axis, var_8);
    	var_12 = df::spatial_transform(var_10, var_11);
    	goto label1;
    }
    var_13 = df::select(var_7, var_2, var_8);
    var_14 = df::select(var_7, var_5, var_12);
    var_15 = df::select(var_7, var_5, var_12);
    var_17 = (var_type == var_16);
    if (var_17) {
    	var_18 = df::add(var_start, var_0);
    	var_19 = df::load(var_joint_q, var_18);
    	var_20 = df::add(var_start, var_6);
    	var_21 = df::load(var_joint_q, var_20);
    	var_22 = df::add(var_start, var_16);
    	var_23 = df::load(var_joint_q, var_22);
    	var_25 = df::add(var_start, var_24);
    	var_26 = df::load(var_joint_q, var_25);
    	var_27 = df::float3(var_9, var_9, var_9);
    	var_28 = df::quat(var_19, var_21, var_23, var_26);
    	var_29 = df::spatial_transform(var_27, var_28);
    	goto label2;
    }
    var_30 = df::select(var_17, var_14, var_29);
    var_31 = df::select(var_17, var_15, var_29);
    var_32 = (var_type == var_24);
    if (var_32) {
    	var_33 = df::spatial_transform_identity();
    	goto label3;
    }
    var_34 = df::select(var_32, var_30, var_33);
    var_35 = df::select(var_32, var_31, var_33);
    var_37 = (var_type == var_36);
    if (var_37) {
    	var_38 = df::add(var_start, var_0);
    	var_39 = df::load(var_joint_q, var_38);
    	var_40 = df::add(var_start, var_6);
    	var_41 = df::load(var_joint_q, var_40);
    	var_42 = df::add(var_start, var_16);
    	var_43 = df::load(var_joint_q, var_42);
    	var_44 = df::add(var_start, var_24);
    	var_45 = df::load(var_joint_q, var_44);
    	var_46 = df::add(var_start, var_36);
    	var_47 = df::load(var_joint_q, var_46);
    	var_49 = df::add(var_start, var_48);
    	var_50 = df::load(var_joint_q, var_49);
    	var_52 = df::add(var_start, var_51);
    	var_53 = df::load(var_joint_q, var_52);
    	var_54 = df::float3(var_39, var_41, var_43);
    	var_55 = df::quat(var_45, var_47, var_50, var_53);
    	var_56 = df::spatial_transform(var_54, var_55);
    	goto label4;
    }
    var_57 = df::select(var_37, var_34, var_56);
    var_58 = df::select(var_37, var_35, var_56);
    var_59 = df::select(var_37, var_19, var_45);
    var_60 = df::select(var_37, var_21, var_47);
    var_61 = df::select(var_37, var_23, var_50);
    var_62 = df::select(var_37, var_26, var_53);
    var_63 = df::spatial_transform_identity();
    goto label5;
    //---------
    // reverse
    label5:;
    adj_63 += adj_ret;
    df::adj_select(var_37, var_26, var_53, adj_37, adj_26, adj_53, adj_62);
    df::adj_select(var_37, var_23, var_50, adj_37, adj_23, adj_50, adj_61);
    df::adj_select(var_37, var_21, var_47, adj_37, adj_21, adj_47, adj_60);
    df::adj_select(var_37, var_19, var_45, adj_37, adj_19, adj_45, adj_59);
    df::adj_select(var_37, var_35, var_56, adj_37, adj_35, adj_56, adj_58);
    df::adj_select(var_37, var_34, var_56, adj_37, adj_34, adj_56, adj_57);
    if (var_37) {
    	label4:;
    	adj_56 += adj_ret;
    	df::adj_spatial_transform(var_54, var_55, adj_54, adj_55, adj_56);
    	df::adj_quat(var_45, var_47, var_50, var_53, adj_45, adj_47, adj_50, adj_53, adj_55);
    	df::adj_float3(var_39, var_41, var_43, adj_39, adj_41, adj_43, adj_54);
    	df::adj_load(var_joint_q, var_52, adj_joint_q, adj_52, adj_53);
    	df::adj_add(var_start, var_51, adj_start, adj_51, adj_52);
    	df::adj_load(var_joint_q, var_49, adj_joint_q, adj_49, adj_50);
    	df::adj_add(var_start, var_48, adj_start, adj_48, adj_49);
    	df::adj_load(var_joint_q, var_46, adj_joint_q, adj_46, adj_47);
    	df::adj_add(var_start, var_36, adj_start, adj_36, adj_46);
    	df::adj_load(var_joint_q, var_44, adj_joint_q, adj_44, adj_45);
    	df::adj_add(var_start, var_24, adj_start, adj_24, adj_44);
    	df::adj_load(var_joint_q, var_42, adj_joint_q, adj_42, adj_43);
    	df::adj_add(var_start, var_16, adj_start, adj_16, adj_42);
    	df::adj_load(var_joint_q, var_40, adj_joint_q, adj_40, adj_41);
    	df::adj_add(var_start, var_6, adj_start, adj_6, adj_40);
    	df::adj_load(var_joint_q, var_38, adj_joint_q, adj_38, adj_39);
    	df::adj_add(var_start, var_0, adj_start, adj_0, adj_38);
    }
    df::adj_select(var_32, var_31, var_33, adj_32, adj_31, adj_33, adj_35);
    df::adj_select(var_32, var_30, var_33, adj_32, adj_30, adj_33, adj_34);
    if (var_32) {
    	label3:;
    	adj_33 += adj_ret;
    }
    df::adj_select(var_17, var_15, var_29, adj_17, adj_15, adj_29, adj_31);
    df::adj_select(var_17, var_14, var_29, adj_17, adj_14, adj_29, adj_30);
    if (var_17) {
    	label2:;
    	adj_29 += adj_ret;
    	df::adj_spatial_transform(var_27, var_28, adj_27, adj_28, adj_29);
    	df::adj_quat(var_19, var_21, var_23, var_26, adj_19, adj_21, adj_23, adj_26, adj_28);
    	df::adj_float3(var_9, var_9, var_9, adj_9, adj_9, adj_9, adj_27);
    	df::adj_load(var_joint_q, var_25, adj_joint_q, adj_25, adj_26);
    	df::adj_add(var_start, var_24, adj_start, adj_24, adj_25);
    	df::adj_load(var_joint_q, var_22, adj_joint_q, adj_22, adj_23);
    	df::adj_add(var_start, var_16, adj_start, adj_16, adj_22);
    	df::adj_load(var_joint_q, var_20, adj_joint_q, adj_20, adj_21);
    	df::adj_add(var_start, var_6, adj_start, adj_6, adj_20);
    	df::adj_load(var_joint_q, var_18, adj_joint_q, adj_18, adj_19);
    	df::adj_add(var_start, var_0, adj_start, adj_0, adj_18);
    }
    df::adj_select(var_7, var_5, var_12, adj_7, adj_5, adj_12, adj_15);
    df::adj_select(var_7, var_5, var_12, adj_7, adj_5, adj_12, adj_14);
    df::adj_select(var_7, var_2, var_8, adj_7, adj_2, adj_8, adj_13);
    if (var_7) {
    	label1:;
    	adj_12 += adj_ret;
    	df::adj_spatial_transform(var_10, var_11, adj_10, adj_11, adj_12);
    	df::adj_quat_from_axis_angle(var_axis, var_8, adj_axis, adj_8, adj_11);
    	df::adj_float3(var_9, var_9, var_9, adj_9, adj_9, adj_9, adj_10);
    	df::adj_load(var_joint_q, var_start, adj_joint_q, adj_start, adj_8);
    }
    if (var_1) {
    	label0:;
    	adj_5 += adj_ret;
    	df::adj_spatial_transform(var_3, var_4, adj_3, adj_4, adj_5);
    	df::adj_mul(var_axis, var_2, adj_axis, adj_2, adj_3);
    	df::adj_load(var_joint_q, var_start, adj_joint_q, adj_start, adj_2);
    }
    return;

}


spatial_vector jcalc_motion_cpu_func(
	int var_type,
	df::float3 var_axis,
	spatial_transform var_X_sc,
	spatial_vector* var_joint_S_s,
	float* var_joint_qd,
	int var_joint_start)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    const float var_2 = 0.0;
    df::float3 var_3;
    spatial_vector var_4;
    spatial_vector var_5;
    float var_6;
    spatial_vector var_7;
    const int var_8 = 1;
    bool var_9;
    df::float3 var_10;
    spatial_vector var_11;
    spatial_vector var_12;
    float var_13;
    spatial_vector var_14;
    spatial_vector var_15;
    spatial_vector var_16;
    spatial_vector var_17;
    const int var_18 = 2;
    bool var_19;
    int var_20;
    float var_21;
    int var_22;
    float var_23;
    int var_24;
    float var_25;
    df::float3 var_26;
    const float var_27 = 1.0;
    spatial_vector var_28;
    spatial_vector var_29;
    spatial_vector var_30;
    spatial_vector var_31;
    spatial_vector var_32;
    spatial_vector var_33;
    int var_34;
    int var_35;
    int var_36;
    float var_37;
    spatial_vector var_38;
    float var_39;
    spatial_vector var_40;
    spatial_vector var_41;
    float var_42;
    spatial_vector var_43;
    spatial_vector var_44;
    spatial_vector var_45;
    const int var_46 = 3;
    bool var_47;
    spatial_vector var_48;
    spatial_vector var_49;
    const int var_50 = 4;
    bool var_51;
    int var_52;
    float var_53;
    int var_54;
    float var_55;
    int var_56;
    float var_57;
    int var_58;
    float var_59;
    int var_60;
    float var_61;
    const int var_62 = 5;
    int var_63;
    float var_64;
    spatial_vector var_65;
    int var_66;
    spatial_vector var_67;
    int var_68;
    spatial_vector var_69;
    int var_70;
    spatial_vector var_71;
    int var_72;
    spatial_vector var_73;
    int var_74;
    spatial_vector var_75;
    int var_76;
    spatial_vector var_77;
    spatial_vector var_78;
    spatial_vector var_79;
    spatial_vector var_80;
    //---------
    // forward
    var_1 = (var_type == var_0);
    if (var_1) {
    	var_3 = df::float3(var_2, var_2, var_2);
    	var_4 = df::spatial_vector(var_3, var_axis);
    	var_5 = spatial_transform_twist_cpu_func(var_X_sc, var_4);
    	var_6 = df::load(var_joint_qd, var_joint_start);
    	var_7 = df::mul(var_5, var_6);
    	df::store(var_joint_S_s, var_joint_start, var_5);
    	return var_7;
    }
    var_9 = (var_type == var_8);
    if (var_9) {
    	var_10 = df::float3(var_2, var_2, var_2);
    	var_11 = df::spatial_vector(var_axis, var_10);
    	var_12 = spatial_transform_twist_cpu_func(var_X_sc, var_11);
    	var_13 = df::load(var_joint_qd, var_joint_start);
    	var_14 = df::mul(var_12, var_13);
    	df::store(var_joint_S_s, var_joint_start, var_12);
    	return var_14;
    }
    var_15 = df::select(var_9, var_5, var_12);
    var_16 = df::select(var_9, var_7, var_14);
    var_17 = df::select(var_9, var_7, var_14);
    var_19 = (var_type == var_18);
    if (var_19) {
    	var_20 = df::add(var_joint_start, var_0);
    	var_21 = df::load(var_joint_qd, var_20);
    	var_22 = df::add(var_joint_start, var_8);
    	var_23 = df::load(var_joint_qd, var_22);
    	var_24 = df::add(var_joint_start, var_18);
    	var_25 = df::load(var_joint_qd, var_24);
    	var_26 = df::float3(var_21, var_23, var_25);
    	var_28 = df::spatial_vector(var_27, var_2, var_2, var_2, var_2, var_2);
    	var_29 = spatial_transform_twist_cpu_func(var_X_sc, var_28);
    	var_30 = df::spatial_vector(var_2, var_27, var_2, var_2, var_2, var_2);
    	var_31 = spatial_transform_twist_cpu_func(var_X_sc, var_30);
    	var_32 = df::spatial_vector(var_2, var_2, var_27, var_2, var_2, var_2);
    	var_33 = spatial_transform_twist_cpu_func(var_X_sc, var_32);
    	var_34 = df::add(var_joint_start, var_0);
    	df::store(var_joint_S_s, var_34, var_29);
    	var_35 = df::add(var_joint_start, var_8);
    	df::store(var_joint_S_s, var_35, var_31);
    	var_36 = df::add(var_joint_start, var_18);
    	df::store(var_joint_S_s, var_36, var_33);
    	var_37 = df::index(var_26, var_0);
    	var_38 = df::mul(var_29, var_37);
    	var_39 = df::index(var_26, var_8);
    	var_40 = df::mul(var_31, var_39);
    	var_41 = df::add(var_38, var_40);
    	var_42 = df::index(var_26, var_18);
    	var_43 = df::mul(var_33, var_42);
    	var_44 = df::add(var_41, var_43);
    	return var_44;
    }
    var_45 = df::select(var_19, var_17, var_44);
    var_47 = (var_type == var_46);
    if (var_47) {
    	var_48 = df::spatial_vector();
    	return var_48;
    }
    var_49 = df::select(var_47, var_45, var_48);
    var_51 = (var_type == var_50);
    if (var_51) {
    	var_52 = df::add(var_joint_start, var_0);
    	var_53 = df::load(var_joint_qd, var_52);
    	var_54 = df::add(var_joint_start, var_8);
    	var_55 = df::load(var_joint_qd, var_54);
    	var_56 = df::add(var_joint_start, var_18);
    	var_57 = df::load(var_joint_qd, var_56);
    	var_58 = df::add(var_joint_start, var_46);
    	var_59 = df::load(var_joint_qd, var_58);
    	var_60 = df::add(var_joint_start, var_50);
    	var_61 = df::load(var_joint_qd, var_60);
    	var_63 = df::add(var_joint_start, var_62);
    	var_64 = df::load(var_joint_qd, var_63);
    	var_65 = df::spatial_vector(var_53, var_55, var_57, var_59, var_61, var_64);
    	var_66 = df::add(var_joint_start, var_0);
    	var_67 = df::spatial_vector(var_27, var_2, var_2, var_2, var_2, var_2);
    	df::store(var_joint_S_s, var_66, var_67);
    	var_68 = df::add(var_joint_start, var_8);
    	var_69 = df::spatial_vector(var_2, var_27, var_2, var_2, var_2, var_2);
    	df::store(var_joint_S_s, var_68, var_69);
    	var_70 = df::add(var_joint_start, var_18);
    	var_71 = df::spatial_vector(var_2, var_2, var_27, var_2, var_2, var_2);
    	df::store(var_joint_S_s, var_70, var_71);
    	var_72 = df::add(var_joint_start, var_46);
    	var_73 = df::spatial_vector(var_2, var_2, var_2, var_27, var_2, var_2);
    	df::store(var_joint_S_s, var_72, var_73);
    	var_74 = df::add(var_joint_start, var_50);
    	var_75 = df::spatial_vector(var_2, var_2, var_2, var_2, var_27, var_2);
    	df::store(var_joint_S_s, var_74, var_75);
    	var_76 = df::add(var_joint_start, var_62);
    	var_77 = df::spatial_vector(var_2, var_2, var_2, var_2, var_2, var_27);
    	df::store(var_joint_S_s, var_76, var_77);
    	return var_65;
    }
    var_78 = df::select(var_51, var_16, var_65);
    var_79 = df::select(var_51, var_49, var_65);
    var_80 = df::spatial_vector();
    return var_80;
    

}

void adj_jcalc_motion_cpu_func(
	int var_type,
	df::float3 var_axis,
	spatial_transform var_X_sc,
	spatial_vector* var_joint_S_s,
	float* var_joint_qd,
	int var_joint_start, 
	int & adj_type,
	df::float3 & adj_axis,
	spatial_transform & adj_X_sc,
	spatial_vector* adj_joint_S_s,
	float* adj_joint_qd,
	int & adj_joint_start,
	spatial_vector & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    const float var_2 = 0.0;
    df::float3 var_3;
    spatial_vector var_4;
    spatial_vector var_5;
    float var_6;
    spatial_vector var_7;
    const int var_8 = 1;
    bool var_9;
    df::float3 var_10;
    spatial_vector var_11;
    spatial_vector var_12;
    float var_13;
    spatial_vector var_14;
    spatial_vector var_15;
    spatial_vector var_16;
    spatial_vector var_17;
    const int var_18 = 2;
    bool var_19;
    int var_20;
    float var_21;
    int var_22;
    float var_23;
    int var_24;
    float var_25;
    df::float3 var_26;
    const float var_27 = 1.0;
    spatial_vector var_28;
    spatial_vector var_29;
    spatial_vector var_30;
    spatial_vector var_31;
    spatial_vector var_32;
    spatial_vector var_33;
    int var_34;
    int var_35;
    int var_36;
    float var_37;
    spatial_vector var_38;
    float var_39;
    spatial_vector var_40;
    spatial_vector var_41;
    float var_42;
    spatial_vector var_43;
    spatial_vector var_44;
    spatial_vector var_45;
    const int var_46 = 3;
    bool var_47;
    spatial_vector var_48;
    spatial_vector var_49;
    const int var_50 = 4;
    bool var_51;
    int var_52;
    float var_53;
    int var_54;
    float var_55;
    int var_56;
    float var_57;
    int var_58;
    float var_59;
    int var_60;
    float var_61;
    const int var_62 = 5;
    int var_63;
    float var_64;
    spatial_vector var_65;
    int var_66;
    spatial_vector var_67;
    int var_68;
    spatial_vector var_69;
    int var_70;
    spatial_vector var_71;
    int var_72;
    spatial_vector var_73;
    int var_74;
    spatial_vector var_75;
    int var_76;
    spatial_vector var_77;
    spatial_vector var_78;
    spatial_vector var_79;
    spatial_vector var_80;
    //---------
    // dual vars
    int adj_0 = 0;
    bool adj_1 = 0;
    float adj_2 = 0;
    df::float3 adj_3 = 0;
    spatial_vector adj_4 = 0;
    spatial_vector adj_5 = 0;
    float adj_6 = 0;
    spatial_vector adj_7 = 0;
    int adj_8 = 0;
    bool adj_9 = 0;
    df::float3 adj_10 = 0;
    spatial_vector adj_11 = 0;
    spatial_vector adj_12 = 0;
    float adj_13 = 0;
    spatial_vector adj_14 = 0;
    spatial_vector adj_15 = 0;
    spatial_vector adj_16 = 0;
    spatial_vector adj_17 = 0;
    int adj_18 = 0;
    bool adj_19 = 0;
    int adj_20 = 0;
    float adj_21 = 0;
    int adj_22 = 0;
    float adj_23 = 0;
    int adj_24 = 0;
    float adj_25 = 0;
    df::float3 adj_26 = 0;
    float adj_27 = 0;
    spatial_vector adj_28 = 0;
    spatial_vector adj_29 = 0;
    spatial_vector adj_30 = 0;
    spatial_vector adj_31 = 0;
    spatial_vector adj_32 = 0;
    spatial_vector adj_33 = 0;
    int adj_34 = 0;
    int adj_35 = 0;
    int adj_36 = 0;
    float adj_37 = 0;
    spatial_vector adj_38 = 0;
    float adj_39 = 0;
    spatial_vector adj_40 = 0;
    spatial_vector adj_41 = 0;
    float adj_42 = 0;
    spatial_vector adj_43 = 0;
    spatial_vector adj_44 = 0;
    spatial_vector adj_45 = 0;
    int adj_46 = 0;
    bool adj_47 = 0;
    spatial_vector adj_48 = 0;
    spatial_vector adj_49 = 0;
    int adj_50 = 0;
    bool adj_51 = 0;
    int adj_52 = 0;
    float adj_53 = 0;
    int adj_54 = 0;
    float adj_55 = 0;
    int adj_56 = 0;
    float adj_57 = 0;
    int adj_58 = 0;
    float adj_59 = 0;
    int adj_60 = 0;
    float adj_61 = 0;
    int adj_62 = 0;
    int adj_63 = 0;
    float adj_64 = 0;
    spatial_vector adj_65 = 0;
    int adj_66 = 0;
    spatial_vector adj_67 = 0;
    int adj_68 = 0;
    spatial_vector adj_69 = 0;
    int adj_70 = 0;
    spatial_vector adj_71 = 0;
    int adj_72 = 0;
    spatial_vector adj_73 = 0;
    int adj_74 = 0;
    spatial_vector adj_75 = 0;
    int adj_76 = 0;
    spatial_vector adj_77 = 0;
    spatial_vector adj_78 = 0;
    spatial_vector adj_79 = 0;
    spatial_vector adj_80 = 0;
    //---------
    // forward
    var_1 = (var_type == var_0);
    if (var_1) {
    	var_3 = df::float3(var_2, var_2, var_2);
    	var_4 = df::spatial_vector(var_3, var_axis);
    	var_5 = spatial_transform_twist_cpu_func(var_X_sc, var_4);
    	var_6 = df::load(var_joint_qd, var_joint_start);
    	var_7 = df::mul(var_5, var_6);
    	df::store(var_joint_S_s, var_joint_start, var_5);
    	goto label0;
    }
    var_9 = (var_type == var_8);
    if (var_9) {
    	var_10 = df::float3(var_2, var_2, var_2);
    	var_11 = df::spatial_vector(var_axis, var_10);
    	var_12 = spatial_transform_twist_cpu_func(var_X_sc, var_11);
    	var_13 = df::load(var_joint_qd, var_joint_start);
    	var_14 = df::mul(var_12, var_13);
    	df::store(var_joint_S_s, var_joint_start, var_12);
    	goto label1;
    }
    var_15 = df::select(var_9, var_5, var_12);
    var_16 = df::select(var_9, var_7, var_14);
    var_17 = df::select(var_9, var_7, var_14);
    var_19 = (var_type == var_18);
    if (var_19) {
    	var_20 = df::add(var_joint_start, var_0);
    	var_21 = df::load(var_joint_qd, var_20);
    	var_22 = df::add(var_joint_start, var_8);
    	var_23 = df::load(var_joint_qd, var_22);
    	var_24 = df::add(var_joint_start, var_18);
    	var_25 = df::load(var_joint_qd, var_24);
    	var_26 = df::float3(var_21, var_23, var_25);
    	var_28 = df::spatial_vector(var_27, var_2, var_2, var_2, var_2, var_2);
    	var_29 = spatial_transform_twist_cpu_func(var_X_sc, var_28);
    	var_30 = df::spatial_vector(var_2, var_27, var_2, var_2, var_2, var_2);
    	var_31 = spatial_transform_twist_cpu_func(var_X_sc, var_30);
    	var_32 = df::spatial_vector(var_2, var_2, var_27, var_2, var_2, var_2);
    	var_33 = spatial_transform_twist_cpu_func(var_X_sc, var_32);
    	var_34 = df::add(var_joint_start, var_0);
    	df::store(var_joint_S_s, var_34, var_29);
    	var_35 = df::add(var_joint_start, var_8);
    	df::store(var_joint_S_s, var_35, var_31);
    	var_36 = df::add(var_joint_start, var_18);
    	df::store(var_joint_S_s, var_36, var_33);
    	var_37 = df::index(var_26, var_0);
    	var_38 = df::mul(var_29, var_37);
    	var_39 = df::index(var_26, var_8);
    	var_40 = df::mul(var_31, var_39);
    	var_41 = df::add(var_38, var_40);
    	var_42 = df::index(var_26, var_18);
    	var_43 = df::mul(var_33, var_42);
    	var_44 = df::add(var_41, var_43);
    	goto label2;
    }
    var_45 = df::select(var_19, var_17, var_44);
    var_47 = (var_type == var_46);
    if (var_47) {
    	var_48 = df::spatial_vector();
    	goto label3;
    }
    var_49 = df::select(var_47, var_45, var_48);
    var_51 = (var_type == var_50);
    if (var_51) {
    	var_52 = df::add(var_joint_start, var_0);
    	var_53 = df::load(var_joint_qd, var_52);
    	var_54 = df::add(var_joint_start, var_8);
    	var_55 = df::load(var_joint_qd, var_54);
    	var_56 = df::add(var_joint_start, var_18);
    	var_57 = df::load(var_joint_qd, var_56);
    	var_58 = df::add(var_joint_start, var_46);
    	var_59 = df::load(var_joint_qd, var_58);
    	var_60 = df::add(var_joint_start, var_50);
    	var_61 = df::load(var_joint_qd, var_60);
    	var_63 = df::add(var_joint_start, var_62);
    	var_64 = df::load(var_joint_qd, var_63);
    	var_65 = df::spatial_vector(var_53, var_55, var_57, var_59, var_61, var_64);
    	var_66 = df::add(var_joint_start, var_0);
    	var_67 = df::spatial_vector(var_27, var_2, var_2, var_2, var_2, var_2);
    	df::store(var_joint_S_s, var_66, var_67);
    	var_68 = df::add(var_joint_start, var_8);
    	var_69 = df::spatial_vector(var_2, var_27, var_2, var_2, var_2, var_2);
    	df::store(var_joint_S_s, var_68, var_69);
    	var_70 = df::add(var_joint_start, var_18);
    	var_71 = df::spatial_vector(var_2, var_2, var_27, var_2, var_2, var_2);
    	df::store(var_joint_S_s, var_70, var_71);
    	var_72 = df::add(var_joint_start, var_46);
    	var_73 = df::spatial_vector(var_2, var_2, var_2, var_27, var_2, var_2);
    	df::store(var_joint_S_s, var_72, var_73);
    	var_74 = df::add(var_joint_start, var_50);
    	var_75 = df::spatial_vector(var_2, var_2, var_2, var_2, var_27, var_2);
    	df::store(var_joint_S_s, var_74, var_75);
    	var_76 = df::add(var_joint_start, var_62);
    	var_77 = df::spatial_vector(var_2, var_2, var_2, var_2, var_2, var_27);
    	df::store(var_joint_S_s, var_76, var_77);
    	goto label4;
    }
    var_78 = df::select(var_51, var_16, var_65);
    var_79 = df::select(var_51, var_49, var_65);
    var_80 = df::spatial_vector();
    goto label5;
    //---------
    // reverse
    label5:;
    adj_80 += adj_ret;
    df::adj_select(var_51, var_49, var_65, adj_51, adj_49, adj_65, adj_79);
    df::adj_select(var_51, var_16, var_65, adj_51, adj_16, adj_65, adj_78);
    if (var_51) {
    	label4:;
    	adj_65 += adj_ret;
    	df::adj_store(var_joint_S_s, var_76, var_77, adj_joint_S_s, adj_76, adj_77);
    	df::adj_spatial_vector(var_2, var_2, var_2, var_2, var_2, var_27, adj_2, adj_2, adj_2, adj_2, adj_2, adj_27, adj_77);
    	df::adj_add(var_joint_start, var_62, adj_joint_start, adj_62, adj_76);
    	df::adj_store(var_joint_S_s, var_74, var_75, adj_joint_S_s, adj_74, adj_75);
    	df::adj_spatial_vector(var_2, var_2, var_2, var_2, var_27, var_2, adj_2, adj_2, adj_2, adj_2, adj_27, adj_2, adj_75);
    	df::adj_add(var_joint_start, var_50, adj_joint_start, adj_50, adj_74);
    	df::adj_store(var_joint_S_s, var_72, var_73, adj_joint_S_s, adj_72, adj_73);
    	df::adj_spatial_vector(var_2, var_2, var_2, var_27, var_2, var_2, adj_2, adj_2, adj_2, adj_27, adj_2, adj_2, adj_73);
    	df::adj_add(var_joint_start, var_46, adj_joint_start, adj_46, adj_72);
    	df::adj_store(var_joint_S_s, var_70, var_71, adj_joint_S_s, adj_70, adj_71);
    	df::adj_spatial_vector(var_2, var_2, var_27, var_2, var_2, var_2, adj_2, adj_2, adj_27, adj_2, adj_2, adj_2, adj_71);
    	df::adj_add(var_joint_start, var_18, adj_joint_start, adj_18, adj_70);
    	df::adj_store(var_joint_S_s, var_68, var_69, adj_joint_S_s, adj_68, adj_69);
    	df::adj_spatial_vector(var_2, var_27, var_2, var_2, var_2, var_2, adj_2, adj_27, adj_2, adj_2, adj_2, adj_2, adj_69);
    	df::adj_add(var_joint_start, var_8, adj_joint_start, adj_8, adj_68);
    	df::adj_store(var_joint_S_s, var_66, var_67, adj_joint_S_s, adj_66, adj_67);
    	df::adj_spatial_vector(var_27, var_2, var_2, var_2, var_2, var_2, adj_27, adj_2, adj_2, adj_2, adj_2, adj_2, adj_67);
    	df::adj_add(var_joint_start, var_0, adj_joint_start, adj_0, adj_66);
    	df::adj_spatial_vector(var_53, var_55, var_57, var_59, var_61, var_64, adj_53, adj_55, adj_57, adj_59, adj_61, adj_64, adj_65);
    	df::adj_load(var_joint_qd, var_63, adj_joint_qd, adj_63, adj_64);
    	df::adj_add(var_joint_start, var_62, adj_joint_start, adj_62, adj_63);
    	df::adj_load(var_joint_qd, var_60, adj_joint_qd, adj_60, adj_61);
    	df::adj_add(var_joint_start, var_50, adj_joint_start, adj_50, adj_60);
    	df::adj_load(var_joint_qd, var_58, adj_joint_qd, adj_58, adj_59);
    	df::adj_add(var_joint_start, var_46, adj_joint_start, adj_46, adj_58);
    	df::adj_load(var_joint_qd, var_56, adj_joint_qd, adj_56, adj_57);
    	df::adj_add(var_joint_start, var_18, adj_joint_start, adj_18, adj_56);
    	df::adj_load(var_joint_qd, var_54, adj_joint_qd, adj_54, adj_55);
    	df::adj_add(var_joint_start, var_8, adj_joint_start, adj_8, adj_54);
    	df::adj_load(var_joint_qd, var_52, adj_joint_qd, adj_52, adj_53);
    	df::adj_add(var_joint_start, var_0, adj_joint_start, adj_0, adj_52);
    }
    df::adj_select(var_47, var_45, var_48, adj_47, adj_45, adj_48, adj_49);
    if (var_47) {
    	label3:;
    	adj_48 += adj_ret;
    }
    df::adj_select(var_19, var_17, var_44, adj_19, adj_17, adj_44, adj_45);
    if (var_19) {
    	label2:;
    	adj_44 += adj_ret;
    	df::adj_add(var_41, var_43, adj_41, adj_43, adj_44);
    	df::adj_mul(var_33, var_42, adj_33, adj_42, adj_43);
    	df::adj_index(var_26, var_18, adj_26, adj_18, adj_42);
    	df::adj_add(var_38, var_40, adj_38, adj_40, adj_41);
    	df::adj_mul(var_31, var_39, adj_31, adj_39, adj_40);
    	df::adj_index(var_26, var_8, adj_26, adj_8, adj_39);
    	df::adj_mul(var_29, var_37, adj_29, adj_37, adj_38);
    	df::adj_index(var_26, var_0, adj_26, adj_0, adj_37);
    	df::adj_store(var_joint_S_s, var_36, var_33, adj_joint_S_s, adj_36, adj_33);
    	df::adj_add(var_joint_start, var_18, adj_joint_start, adj_18, adj_36);
    	df::adj_store(var_joint_S_s, var_35, var_31, adj_joint_S_s, adj_35, adj_31);
    	df::adj_add(var_joint_start, var_8, adj_joint_start, adj_8, adj_35);
    	df::adj_store(var_joint_S_s, var_34, var_29, adj_joint_S_s, adj_34, adj_29);
    	df::adj_add(var_joint_start, var_0, adj_joint_start, adj_0, adj_34);
    	adj_spatial_transform_twist_cpu_func(var_X_sc, var_32, adj_X_sc, adj_32, adj_33);
    	df::adj_spatial_vector(var_2, var_2, var_27, var_2, var_2, var_2, adj_2, adj_2, adj_27, adj_2, adj_2, adj_2, adj_32);
    	adj_spatial_transform_twist_cpu_func(var_X_sc, var_30, adj_X_sc, adj_30, adj_31);
    	df::adj_spatial_vector(var_2, var_27, var_2, var_2, var_2, var_2, adj_2, adj_27, adj_2, adj_2, adj_2, adj_2, adj_30);
    	adj_spatial_transform_twist_cpu_func(var_X_sc, var_28, adj_X_sc, adj_28, adj_29);
    	df::adj_spatial_vector(var_27, var_2, var_2, var_2, var_2, var_2, adj_27, adj_2, adj_2, adj_2, adj_2, adj_2, adj_28);
    	df::adj_float3(var_21, var_23, var_25, adj_21, adj_23, adj_25, adj_26);
    	df::adj_load(var_joint_qd, var_24, adj_joint_qd, adj_24, adj_25);
    	df::adj_add(var_joint_start, var_18, adj_joint_start, adj_18, adj_24);
    	df::adj_load(var_joint_qd, var_22, adj_joint_qd, adj_22, adj_23);
    	df::adj_add(var_joint_start, var_8, adj_joint_start, adj_8, adj_22);
    	df::adj_load(var_joint_qd, var_20, adj_joint_qd, adj_20, adj_21);
    	df::adj_add(var_joint_start, var_0, adj_joint_start, adj_0, adj_20);
    }
    df::adj_select(var_9, var_7, var_14, adj_9, adj_7, adj_14, adj_17);
    df::adj_select(var_9, var_7, var_14, adj_9, adj_7, adj_14, adj_16);
    df::adj_select(var_9, var_5, var_12, adj_9, adj_5, adj_12, adj_15);
    if (var_9) {
    	label1:;
    	adj_14 += adj_ret;
    	df::adj_store(var_joint_S_s, var_joint_start, var_12, adj_joint_S_s, adj_joint_start, adj_12);
    	df::adj_mul(var_12, var_13, adj_12, adj_13, adj_14);
    	df::adj_load(var_joint_qd, var_joint_start, adj_joint_qd, adj_joint_start, adj_13);
    	adj_spatial_transform_twist_cpu_func(var_X_sc, var_11, adj_X_sc, adj_11, adj_12);
    	df::adj_spatial_vector(var_axis, var_10, adj_axis, adj_10, adj_11);
    	df::adj_float3(var_2, var_2, var_2, adj_2, adj_2, adj_2, adj_10);
    }
    if (var_1) {
    	label0:;
    	adj_7 += adj_ret;
    	df::adj_store(var_joint_S_s, var_joint_start, var_5, adj_joint_S_s, adj_joint_start, adj_5);
    	df::adj_mul(var_5, var_6, adj_5, adj_6, adj_7);
    	df::adj_load(var_joint_qd, var_joint_start, adj_joint_qd, adj_joint_start, adj_6);
    	adj_spatial_transform_twist_cpu_func(var_X_sc, var_4, adj_X_sc, adj_4, adj_5);
    	df::adj_spatial_vector(var_3, var_axis, adj_3, adj_axis, adj_4);
    	df::adj_float3(var_2, var_2, var_2, adj_2, adj_2, adj_2, adj_3);
    }
    return;

}


int jcalc_tau_cpu_func(
	int var_type,
	float var_target_k_e,
	float var_target_k_d,
	float var_limit_k_e,
	float var_limit_k_d,
	spatial_vector* var_joint_S_s,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_act,
	float* var_joint_target,
	float* var_joint_limit_lower,
	float* var_joint_limit_upper,
	int var_coord_start,
	int var_dof_start,
	spatial_vector var_body_f_s,
	float* var_tau)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    const int var_2 = 1;
    bool var_3;
    bool var_4;
    spatial_vector var_5;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    float var_10;
    float var_11;
    const float var_12 = 0.0;
    bool var_13;
    float var_14;
    float var_15;
    float var_16;
    bool var_17;
    float var_18;
    float var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    float var_32;
    const int var_33 = 2;
    bool var_34;
    int var_35;
    float var_36;
    int var_37;
    float var_38;
    int var_39;
    float var_40;
    df::float3 var_41;
    int var_42;
    float var_43;
    int var_44;
    float var_45;
    int var_46;
    float var_47;
    df::float3 var_48;
    const int var_49 = 0;
    int var_50;
    spatial_vector var_51;
    float var_52;
    float var_53;
    int var_54;
    float var_55;
    float var_56;
    float var_57;
    float var_58;
    float var_59;
    float var_60;
    const int var_61 = 1;
    int var_62;
    spatial_vector var_63;
    float var_64;
    float var_65;
    int var_66;
    float var_67;
    float var_68;
    float var_69;
    float var_70;
    float var_71;
    float var_72;
    const int var_73 = 2;
    int var_74;
    spatial_vector var_75;
    float var_76;
    float var_77;
    int var_78;
    float var_79;
    float var_80;
    float var_81;
    float var_82;
    float var_83;
    float var_84;
    spatial_vector var_85;
    const int var_86 = 4;
    bool var_87;
    const int var_88 = 0;
    int var_89;
    spatial_vector var_90;
    int var_91;
    float var_92;
    float var_93;
    const int var_94 = 1;
    int var_95;
    spatial_vector var_96;
    int var_97;
    float var_98;
    float var_99;
    const int var_100 = 2;
    int var_101;
    spatial_vector var_102;
    int var_103;
    float var_104;
    float var_105;
    const int var_106 = 3;
    int var_107;
    spatial_vector var_108;
    int var_109;
    float var_110;
    float var_111;
    const int var_112 = 4;
    int var_113;
    spatial_vector var_114;
    int var_115;
    float var_116;
    float var_117;
    const int var_118 = 5;
    int var_119;
    spatial_vector var_120;
    int var_121;
    float var_122;
    float var_123;
    spatial_vector var_124;
    int var_125;
    //---------
    // forward
    var_1 = (var_type == var_0);
    var_3 = (var_type == var_2);
    var_4 = var_1 || var_3;
    if (var_4) {
    	var_5 = df::load(var_joint_S_s, var_dof_start);
    	var_6 = df::load(var_joint_q, var_coord_start);
    	var_7 = df::load(var_joint_qd, var_dof_start);
    	var_8 = df::load(var_joint_act, var_dof_start);
    	var_9 = df::load(var_joint_target, var_coord_start);
    	var_10 = df::load(var_joint_limit_lower, var_coord_start);
    	var_11 = df::load(var_joint_limit_upper, var_coord_start);
    	var_13 = (var_6 < var_10);
    	if (var_13) {
    		var_14 = df::sub(var_10, var_6);
    		var_15 = df::mul(var_limit_k_e, var_14);
    	}
    	var_16 = df::select(var_13, var_12, var_15);
    	var_17 = (var_6 > var_11);
    	if (var_17) {
    		var_18 = df::sub(var_11, var_6);
    		var_19 = df::mul(var_limit_k_e, var_18);
    	}
    	var_20 = df::select(var_17, var_16, var_19);
    	var_21 = df::sub(var_12, var_limit_k_d);
    	var_22 = df::mul(var_21, var_7);
    	var_23 = df::spatial_dot(var_5, var_body_f_s);
    	var_24 = df::sub(var_12, var_23);
    	var_25 = df::sub(var_6, var_9);
    	var_26 = df::mul(var_target_k_e, var_25);
    	var_27 = df::sub(var_24, var_26);
    	var_28 = df::mul(var_target_k_d, var_7);
    	var_29 = df::sub(var_27, var_28);
    	var_30 = df::add(var_29, var_8);
    	var_31 = df::add(var_30, var_20);
    	var_32 = df::add(var_31, var_22);
    	df::store(var_tau, var_dof_start, var_32);
    }
    var_34 = (var_type == var_33);
    if (var_34) {
    	var_35 = df::add(var_coord_start, var_0);
    	var_36 = df::load(var_joint_q, var_35);
    	var_37 = df::add(var_coord_start, var_2);
    	var_38 = df::load(var_joint_q, var_37);
    	var_39 = df::add(var_coord_start, var_33);
    	var_40 = df::load(var_joint_q, var_39);
    	var_41 = df::float3(var_36, var_38, var_40);
    	var_42 = df::add(var_dof_start, var_0);
    	var_43 = df::load(var_joint_qd, var_42);
    	var_44 = df::add(var_dof_start, var_2);
    	var_45 = df::load(var_joint_qd, var_44);
    	var_46 = df::add(var_dof_start, var_33);
    	var_47 = df::load(var_joint_qd, var_46);
    	var_48 = df::float3(var_43, var_45, var_47);
    	var_50 = df::add(var_dof_start, var_49);
    	var_51 = df::load(var_joint_S_s, var_50);
    	var_52 = df::index(var_48, var_49);
    	var_53 = df::index(var_41, var_49);
    	var_54 = df::add(var_dof_start, var_49);
    	var_55 = df::spatial_dot(var_51, var_body_f_s);
    	var_56 = df::sub(var_12, var_55);
    	var_57 = df::mul(var_52, var_target_k_d);
    	var_58 = df::sub(var_56, var_57);
    	var_59 = df::mul(var_53, var_target_k_e);
    	var_60 = df::sub(var_58, var_59);
    	df::store(var_tau, var_54, var_60);
    	var_62 = df::add(var_dof_start, var_61);
    	var_63 = df::load(var_joint_S_s, var_62);
    	var_64 = df::index(var_48, var_61);
    	var_65 = df::index(var_41, var_61);
    	var_66 = df::add(var_dof_start, var_61);
    	var_67 = df::spatial_dot(var_63, var_body_f_s);
    	var_68 = df::sub(var_12, var_67);
    	var_69 = df::mul(var_64, var_target_k_d);
    	var_70 = df::sub(var_68, var_69);
    	var_71 = df::mul(var_65, var_target_k_e);
    	var_72 = df::sub(var_70, var_71);
    	df::store(var_tau, var_66, var_72);
    	var_74 = df::add(var_dof_start, var_73);
    	var_75 = df::load(var_joint_S_s, var_74);
    	var_76 = df::index(var_48, var_73);
    	var_77 = df::index(var_41, var_73);
    	var_78 = df::add(var_dof_start, var_73);
    	var_79 = df::spatial_dot(var_75, var_body_f_s);
    	var_80 = df::sub(var_12, var_79);
    	var_81 = df::mul(var_76, var_target_k_d);
    	var_82 = df::sub(var_80, var_81);
    	var_83 = df::mul(var_77, var_target_k_e);
    	var_84 = df::sub(var_82, var_83);
    	df::store(var_tau, var_78, var_84);
    }
    var_85 = df::select(var_34, var_5, var_75);
    var_87 = (var_type == var_86);
    if (var_87) {
    	var_89 = df::add(var_dof_start, var_88);
    	var_90 = df::load(var_joint_S_s, var_89);
    	var_91 = df::add(var_dof_start, var_88);
    	var_92 = df::spatial_dot(var_90, var_body_f_s);
    	var_93 = df::sub(var_12, var_92);
    	df::store(var_tau, var_91, var_93);
    	var_95 = df::add(var_dof_start, var_94);
    	var_96 = df::load(var_joint_S_s, var_95);
    	var_97 = df::add(var_dof_start, var_94);
    	var_98 = df::spatial_dot(var_96, var_body_f_s);
    	var_99 = df::sub(var_12, var_98);
    	df::store(var_tau, var_97, var_99);
    	var_101 = df::add(var_dof_start, var_100);
    	var_102 = df::load(var_joint_S_s, var_101);
    	var_103 = df::add(var_dof_start, var_100);
    	var_104 = df::spatial_dot(var_102, var_body_f_s);
    	var_105 = df::sub(var_12, var_104);
    	df::store(var_tau, var_103, var_105);
    	var_107 = df::add(var_dof_start, var_106);
    	var_108 = df::load(var_joint_S_s, var_107);
    	var_109 = df::add(var_dof_start, var_106);
    	var_110 = df::spatial_dot(var_108, var_body_f_s);
    	var_111 = df::sub(var_12, var_110);
    	df::store(var_tau, var_109, var_111);
    	var_113 = df::add(var_dof_start, var_112);
    	var_114 = df::load(var_joint_S_s, var_113);
    	var_115 = df::add(var_dof_start, var_112);
    	var_116 = df::spatial_dot(var_114, var_body_f_s);
    	var_117 = df::sub(var_12, var_116);
    	df::store(var_tau, var_115, var_117);
    	var_119 = df::add(var_dof_start, var_118);
    	var_120 = df::load(var_joint_S_s, var_119);
    	var_121 = df::add(var_dof_start, var_118);
    	var_122 = df::spatial_dot(var_120, var_body_f_s);
    	var_123 = df::sub(var_12, var_122);
    	df::store(var_tau, var_121, var_123);
    }
    var_124 = df::select(var_87, var_85, var_120);
    var_125 = df::select(var_87, var_73, var_118);
    return var_0;
    

}

void adj_jcalc_tau_cpu_func(
	int var_type,
	float var_target_k_e,
	float var_target_k_d,
	float var_limit_k_e,
	float var_limit_k_d,
	spatial_vector* var_joint_S_s,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_act,
	float* var_joint_target,
	float* var_joint_limit_lower,
	float* var_joint_limit_upper,
	int var_coord_start,
	int var_dof_start,
	spatial_vector var_body_f_s,
	float* var_tau, 
	int & adj_type,
	float & adj_target_k_e,
	float & adj_target_k_d,
	float & adj_limit_k_e,
	float & adj_limit_k_d,
	spatial_vector* adj_joint_S_s,
	float* adj_joint_q,
	float* adj_joint_qd,
	float* adj_joint_act,
	float* adj_joint_target,
	float* adj_joint_limit_lower,
	float* adj_joint_limit_upper,
	int & adj_coord_start,
	int & adj_dof_start,
	spatial_vector & adj_body_f_s,
	float* adj_tau,
	int & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    const int var_2 = 1;
    bool var_3;
    bool var_4;
    spatial_vector var_5;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    float var_10;
    float var_11;
    const float var_12 = 0.0;
    bool var_13;
    float var_14;
    float var_15;
    float var_16;
    bool var_17;
    float var_18;
    float var_19;
    float var_20;
    float var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    float var_32;
    const int var_33 = 2;
    bool var_34;
    int var_35;
    float var_36;
    int var_37;
    float var_38;
    int var_39;
    float var_40;
    df::float3 var_41;
    int var_42;
    float var_43;
    int var_44;
    float var_45;
    int var_46;
    float var_47;
    df::float3 var_48;
    const int var_49 = 0;
    int var_50;
    spatial_vector var_51;
    float var_52;
    float var_53;
    int var_54;
    float var_55;
    float var_56;
    float var_57;
    float var_58;
    float var_59;
    float var_60;
    const int var_61 = 1;
    int var_62;
    spatial_vector var_63;
    float var_64;
    float var_65;
    int var_66;
    float var_67;
    float var_68;
    float var_69;
    float var_70;
    float var_71;
    float var_72;
    const int var_73 = 2;
    int var_74;
    spatial_vector var_75;
    float var_76;
    float var_77;
    int var_78;
    float var_79;
    float var_80;
    float var_81;
    float var_82;
    float var_83;
    float var_84;
    spatial_vector var_85;
    const int var_86 = 4;
    bool var_87;
    const int var_88 = 0;
    int var_89;
    spatial_vector var_90;
    int var_91;
    float var_92;
    float var_93;
    const int var_94 = 1;
    int var_95;
    spatial_vector var_96;
    int var_97;
    float var_98;
    float var_99;
    const int var_100 = 2;
    int var_101;
    spatial_vector var_102;
    int var_103;
    float var_104;
    float var_105;
    const int var_106 = 3;
    int var_107;
    spatial_vector var_108;
    int var_109;
    float var_110;
    float var_111;
    const int var_112 = 4;
    int var_113;
    spatial_vector var_114;
    int var_115;
    float var_116;
    float var_117;
    const int var_118 = 5;
    int var_119;
    spatial_vector var_120;
    int var_121;
    float var_122;
    float var_123;
    spatial_vector var_124;
    int var_125;
    //---------
    // dual vars
    int adj_0 = 0;
    bool adj_1 = 0;
    int adj_2 = 0;
    bool adj_3 = 0;
    bool adj_4 = 0;
    spatial_vector adj_5 = 0;
    float adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    float adj_10 = 0;
    float adj_11 = 0;
    float adj_12 = 0;
    bool adj_13 = 0;
    float adj_14 = 0;
    float adj_15 = 0;
    float adj_16 = 0;
    bool adj_17 = 0;
    float adj_18 = 0;
    float adj_19 = 0;
    float adj_20 = 0;
    float adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    float adj_26 = 0;
    float adj_27 = 0;
    float adj_28 = 0;
    float adj_29 = 0;
    float adj_30 = 0;
    float adj_31 = 0;
    float adj_32 = 0;
    int adj_33 = 0;
    bool adj_34 = 0;
    int adj_35 = 0;
    float adj_36 = 0;
    int adj_37 = 0;
    float adj_38 = 0;
    int adj_39 = 0;
    float adj_40 = 0;
    df::float3 adj_41 = 0;
    int adj_42 = 0;
    float adj_43 = 0;
    int adj_44 = 0;
    float adj_45 = 0;
    int adj_46 = 0;
    float adj_47 = 0;
    df::float3 adj_48 = 0;
    int adj_49 = 0;
    int adj_50 = 0;
    spatial_vector adj_51 = 0;
    float adj_52 = 0;
    float adj_53 = 0;
    int adj_54 = 0;
    float adj_55 = 0;
    float adj_56 = 0;
    float adj_57 = 0;
    float adj_58 = 0;
    float adj_59 = 0;
    float adj_60 = 0;
    int adj_61 = 0;
    int adj_62 = 0;
    spatial_vector adj_63 = 0;
    float adj_64 = 0;
    float adj_65 = 0;
    int adj_66 = 0;
    float adj_67 = 0;
    float adj_68 = 0;
    float adj_69 = 0;
    float adj_70 = 0;
    float adj_71 = 0;
    float adj_72 = 0;
    int adj_73 = 0;
    int adj_74 = 0;
    spatial_vector adj_75 = 0;
    float adj_76 = 0;
    float adj_77 = 0;
    int adj_78 = 0;
    float adj_79 = 0;
    float adj_80 = 0;
    float adj_81 = 0;
    float adj_82 = 0;
    float adj_83 = 0;
    float adj_84 = 0;
    spatial_vector adj_85 = 0;
    int adj_86 = 0;
    bool adj_87 = 0;
    int adj_88 = 0;
    int adj_89 = 0;
    spatial_vector adj_90 = 0;
    int adj_91 = 0;
    float adj_92 = 0;
    float adj_93 = 0;
    int adj_94 = 0;
    int adj_95 = 0;
    spatial_vector adj_96 = 0;
    int adj_97 = 0;
    float adj_98 = 0;
    float adj_99 = 0;
    int adj_100 = 0;
    int adj_101 = 0;
    spatial_vector adj_102 = 0;
    int adj_103 = 0;
    float adj_104 = 0;
    float adj_105 = 0;
    int adj_106 = 0;
    int adj_107 = 0;
    spatial_vector adj_108 = 0;
    int adj_109 = 0;
    float adj_110 = 0;
    float adj_111 = 0;
    int adj_112 = 0;
    int adj_113 = 0;
    spatial_vector adj_114 = 0;
    int adj_115 = 0;
    float adj_116 = 0;
    float adj_117 = 0;
    int adj_118 = 0;
    int adj_119 = 0;
    spatial_vector adj_120 = 0;
    int adj_121 = 0;
    float adj_122 = 0;
    float adj_123 = 0;
    spatial_vector adj_124 = 0;
    int adj_125 = 0;
    //---------
    // forward
    var_1 = (var_type == var_0);
    var_3 = (var_type == var_2);
    var_4 = var_1 || var_3;
    if (var_4) {
    	var_5 = df::load(var_joint_S_s, var_dof_start);
    	var_6 = df::load(var_joint_q, var_coord_start);
    	var_7 = df::load(var_joint_qd, var_dof_start);
    	var_8 = df::load(var_joint_act, var_dof_start);
    	var_9 = df::load(var_joint_target, var_coord_start);
    	var_10 = df::load(var_joint_limit_lower, var_coord_start);
    	var_11 = df::load(var_joint_limit_upper, var_coord_start);
    	var_13 = (var_6 < var_10);
    	if (var_13) {
    		var_14 = df::sub(var_10, var_6);
    		var_15 = df::mul(var_limit_k_e, var_14);
    	}
    	var_16 = df::select(var_13, var_12, var_15);
    	var_17 = (var_6 > var_11);
    	if (var_17) {
    		var_18 = df::sub(var_11, var_6);
    		var_19 = df::mul(var_limit_k_e, var_18);
    	}
    	var_20 = df::select(var_17, var_16, var_19);
    	var_21 = df::sub(var_12, var_limit_k_d);
    	var_22 = df::mul(var_21, var_7);
    	var_23 = df::spatial_dot(var_5, var_body_f_s);
    	var_24 = df::sub(var_12, var_23);
    	var_25 = df::sub(var_6, var_9);
    	var_26 = df::mul(var_target_k_e, var_25);
    	var_27 = df::sub(var_24, var_26);
    	var_28 = df::mul(var_target_k_d, var_7);
    	var_29 = df::sub(var_27, var_28);
    	var_30 = df::add(var_29, var_8);
    	var_31 = df::add(var_30, var_20);
    	var_32 = df::add(var_31, var_22);
    	df::store(var_tau, var_dof_start, var_32);
    }
    var_34 = (var_type == var_33);
    if (var_34) {
    	var_35 = df::add(var_coord_start, var_0);
    	var_36 = df::load(var_joint_q, var_35);
    	var_37 = df::add(var_coord_start, var_2);
    	var_38 = df::load(var_joint_q, var_37);
    	var_39 = df::add(var_coord_start, var_33);
    	var_40 = df::load(var_joint_q, var_39);
    	var_41 = df::float3(var_36, var_38, var_40);
    	var_42 = df::add(var_dof_start, var_0);
    	var_43 = df::load(var_joint_qd, var_42);
    	var_44 = df::add(var_dof_start, var_2);
    	var_45 = df::load(var_joint_qd, var_44);
    	var_46 = df::add(var_dof_start, var_33);
    	var_47 = df::load(var_joint_qd, var_46);
    	var_48 = df::float3(var_43, var_45, var_47);
    	var_50 = df::add(var_dof_start, var_49);
    	var_51 = df::load(var_joint_S_s, var_50);
    	var_52 = df::index(var_48, var_49);
    	var_53 = df::index(var_41, var_49);
    	var_54 = df::add(var_dof_start, var_49);
    	var_55 = df::spatial_dot(var_51, var_body_f_s);
    	var_56 = df::sub(var_12, var_55);
    	var_57 = df::mul(var_52, var_target_k_d);
    	var_58 = df::sub(var_56, var_57);
    	var_59 = df::mul(var_53, var_target_k_e);
    	var_60 = df::sub(var_58, var_59);
    	df::store(var_tau, var_54, var_60);
    	var_62 = df::add(var_dof_start, var_61);
    	var_63 = df::load(var_joint_S_s, var_62);
    	var_64 = df::index(var_48, var_61);
    	var_65 = df::index(var_41, var_61);
    	var_66 = df::add(var_dof_start, var_61);
    	var_67 = df::spatial_dot(var_63, var_body_f_s);
    	var_68 = df::sub(var_12, var_67);
    	var_69 = df::mul(var_64, var_target_k_d);
    	var_70 = df::sub(var_68, var_69);
    	var_71 = df::mul(var_65, var_target_k_e);
    	var_72 = df::sub(var_70, var_71);
    	df::store(var_tau, var_66, var_72);
    	var_74 = df::add(var_dof_start, var_73);
    	var_75 = df::load(var_joint_S_s, var_74);
    	var_76 = df::index(var_48, var_73);
    	var_77 = df::index(var_41, var_73);
    	var_78 = df::add(var_dof_start, var_73);
    	var_79 = df::spatial_dot(var_75, var_body_f_s);
    	var_80 = df::sub(var_12, var_79);
    	var_81 = df::mul(var_76, var_target_k_d);
    	var_82 = df::sub(var_80, var_81);
    	var_83 = df::mul(var_77, var_target_k_e);
    	var_84 = df::sub(var_82, var_83);
    	df::store(var_tau, var_78, var_84);
    }
    var_85 = df::select(var_34, var_5, var_75);
    var_87 = (var_type == var_86);
    if (var_87) {
    	var_89 = df::add(var_dof_start, var_88);
    	var_90 = df::load(var_joint_S_s, var_89);
    	var_91 = df::add(var_dof_start, var_88);
    	var_92 = df::spatial_dot(var_90, var_body_f_s);
    	var_93 = df::sub(var_12, var_92);
    	df::store(var_tau, var_91, var_93);
    	var_95 = df::add(var_dof_start, var_94);
    	var_96 = df::load(var_joint_S_s, var_95);
    	var_97 = df::add(var_dof_start, var_94);
    	var_98 = df::spatial_dot(var_96, var_body_f_s);
    	var_99 = df::sub(var_12, var_98);
    	df::store(var_tau, var_97, var_99);
    	var_101 = df::add(var_dof_start, var_100);
    	var_102 = df::load(var_joint_S_s, var_101);
    	var_103 = df::add(var_dof_start, var_100);
    	var_104 = df::spatial_dot(var_102, var_body_f_s);
    	var_105 = df::sub(var_12, var_104);
    	df::store(var_tau, var_103, var_105);
    	var_107 = df::add(var_dof_start, var_106);
    	var_108 = df::load(var_joint_S_s, var_107);
    	var_109 = df::add(var_dof_start, var_106);
    	var_110 = df::spatial_dot(var_108, var_body_f_s);
    	var_111 = df::sub(var_12, var_110);
    	df::store(var_tau, var_109, var_111);
    	var_113 = df::add(var_dof_start, var_112);
    	var_114 = df::load(var_joint_S_s, var_113);
    	var_115 = df::add(var_dof_start, var_112);
    	var_116 = df::spatial_dot(var_114, var_body_f_s);
    	var_117 = df::sub(var_12, var_116);
    	df::store(var_tau, var_115, var_117);
    	var_119 = df::add(var_dof_start, var_118);
    	var_120 = df::load(var_joint_S_s, var_119);
    	var_121 = df::add(var_dof_start, var_118);
    	var_122 = df::spatial_dot(var_120, var_body_f_s);
    	var_123 = df::sub(var_12, var_122);
    	df::store(var_tau, var_121, var_123);
    }
    var_124 = df::select(var_87, var_85, var_120);
    var_125 = df::select(var_87, var_73, var_118);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_0 += adj_ret;
    df::adj_select(var_87, var_73, var_118, adj_87, adj_73, adj_118, adj_125);
    df::adj_select(var_87, var_85, var_120, adj_87, adj_85, adj_120, adj_124);
    if (var_87) {
    	df::adj_store(var_tau, var_121, var_123, adj_tau, adj_121, adj_123);
    	df::adj_sub(var_12, var_122, adj_12, adj_122, adj_123);
    	df::adj_spatial_dot(var_120, var_body_f_s, adj_120, adj_body_f_s, adj_122);
    	df::adj_add(var_dof_start, var_118, adj_dof_start, adj_118, adj_121);
    	df::adj_load(var_joint_S_s, var_119, adj_joint_S_s, adj_119, adj_120);
    	df::adj_add(var_dof_start, var_118, adj_dof_start, adj_118, adj_119);
    	df::adj_store(var_tau, var_115, var_117, adj_tau, adj_115, adj_117);
    	df::adj_sub(var_12, var_116, adj_12, adj_116, adj_117);
    	df::adj_spatial_dot(var_114, var_body_f_s, adj_114, adj_body_f_s, adj_116);
    	df::adj_add(var_dof_start, var_112, adj_dof_start, adj_112, adj_115);
    	df::adj_load(var_joint_S_s, var_113, adj_joint_S_s, adj_113, adj_114);
    	df::adj_add(var_dof_start, var_112, adj_dof_start, adj_112, adj_113);
    	df::adj_store(var_tau, var_109, var_111, adj_tau, adj_109, adj_111);
    	df::adj_sub(var_12, var_110, adj_12, adj_110, adj_111);
    	df::adj_spatial_dot(var_108, var_body_f_s, adj_108, adj_body_f_s, adj_110);
    	df::adj_add(var_dof_start, var_106, adj_dof_start, adj_106, adj_109);
    	df::adj_load(var_joint_S_s, var_107, adj_joint_S_s, adj_107, adj_108);
    	df::adj_add(var_dof_start, var_106, adj_dof_start, adj_106, adj_107);
    	df::adj_store(var_tau, var_103, var_105, adj_tau, adj_103, adj_105);
    	df::adj_sub(var_12, var_104, adj_12, adj_104, adj_105);
    	df::adj_spatial_dot(var_102, var_body_f_s, adj_102, adj_body_f_s, adj_104);
    	df::adj_add(var_dof_start, var_100, adj_dof_start, adj_100, adj_103);
    	df::adj_load(var_joint_S_s, var_101, adj_joint_S_s, adj_101, adj_102);
    	df::adj_add(var_dof_start, var_100, adj_dof_start, adj_100, adj_101);
    	df::adj_store(var_tau, var_97, var_99, adj_tau, adj_97, adj_99);
    	df::adj_sub(var_12, var_98, adj_12, adj_98, adj_99);
    	df::adj_spatial_dot(var_96, var_body_f_s, adj_96, adj_body_f_s, adj_98);
    	df::adj_add(var_dof_start, var_94, adj_dof_start, adj_94, adj_97);
    	df::adj_load(var_joint_S_s, var_95, adj_joint_S_s, adj_95, adj_96);
    	df::adj_add(var_dof_start, var_94, adj_dof_start, adj_94, adj_95);
    	df::adj_store(var_tau, var_91, var_93, adj_tau, adj_91, adj_93);
    	df::adj_sub(var_12, var_92, adj_12, adj_92, adj_93);
    	df::adj_spatial_dot(var_90, var_body_f_s, adj_90, adj_body_f_s, adj_92);
    	df::adj_add(var_dof_start, var_88, adj_dof_start, adj_88, adj_91);
    	df::adj_load(var_joint_S_s, var_89, adj_joint_S_s, adj_89, adj_90);
    	df::adj_add(var_dof_start, var_88, adj_dof_start, adj_88, adj_89);
    }
    df::adj_select(var_34, var_5, var_75, adj_34, adj_5, adj_75, adj_85);
    if (var_34) {
    	df::adj_store(var_tau, var_78, var_84, adj_tau, adj_78, adj_84);
    	df::adj_sub(var_82, var_83, adj_82, adj_83, adj_84);
    	df::adj_mul(var_77, var_target_k_e, adj_77, adj_target_k_e, adj_83);
    	df::adj_sub(var_80, var_81, adj_80, adj_81, adj_82);
    	df::adj_mul(var_76, var_target_k_d, adj_76, adj_target_k_d, adj_81);
    	df::adj_sub(var_12, var_79, adj_12, adj_79, adj_80);
    	df::adj_spatial_dot(var_75, var_body_f_s, adj_75, adj_body_f_s, adj_79);
    	df::adj_add(var_dof_start, var_73, adj_dof_start, adj_73, adj_78);
    	df::adj_index(var_41, var_73, adj_41, adj_73, adj_77);
    	df::adj_index(var_48, var_73, adj_48, adj_73, adj_76);
    	df::adj_load(var_joint_S_s, var_74, adj_joint_S_s, adj_74, adj_75);
    	df::adj_add(var_dof_start, var_73, adj_dof_start, adj_73, adj_74);
    	df::adj_store(var_tau, var_66, var_72, adj_tau, adj_66, adj_72);
    	df::adj_sub(var_70, var_71, adj_70, adj_71, adj_72);
    	df::adj_mul(var_65, var_target_k_e, adj_65, adj_target_k_e, adj_71);
    	df::adj_sub(var_68, var_69, adj_68, adj_69, adj_70);
    	df::adj_mul(var_64, var_target_k_d, adj_64, adj_target_k_d, adj_69);
    	df::adj_sub(var_12, var_67, adj_12, adj_67, adj_68);
    	df::adj_spatial_dot(var_63, var_body_f_s, adj_63, adj_body_f_s, adj_67);
    	df::adj_add(var_dof_start, var_61, adj_dof_start, adj_61, adj_66);
    	df::adj_index(var_41, var_61, adj_41, adj_61, adj_65);
    	df::adj_index(var_48, var_61, adj_48, adj_61, adj_64);
    	df::adj_load(var_joint_S_s, var_62, adj_joint_S_s, adj_62, adj_63);
    	df::adj_add(var_dof_start, var_61, adj_dof_start, adj_61, adj_62);
    	df::adj_store(var_tau, var_54, var_60, adj_tau, adj_54, adj_60);
    	df::adj_sub(var_58, var_59, adj_58, adj_59, adj_60);
    	df::adj_mul(var_53, var_target_k_e, adj_53, adj_target_k_e, adj_59);
    	df::adj_sub(var_56, var_57, adj_56, adj_57, adj_58);
    	df::adj_mul(var_52, var_target_k_d, adj_52, adj_target_k_d, adj_57);
    	df::adj_sub(var_12, var_55, adj_12, adj_55, adj_56);
    	df::adj_spatial_dot(var_51, var_body_f_s, adj_51, adj_body_f_s, adj_55);
    	df::adj_add(var_dof_start, var_49, adj_dof_start, adj_49, adj_54);
    	df::adj_index(var_41, var_49, adj_41, adj_49, adj_53);
    	df::adj_index(var_48, var_49, adj_48, adj_49, adj_52);
    	df::adj_load(var_joint_S_s, var_50, adj_joint_S_s, adj_50, adj_51);
    	df::adj_add(var_dof_start, var_49, adj_dof_start, adj_49, adj_50);
    	df::adj_float3(var_43, var_45, var_47, adj_43, adj_45, adj_47, adj_48);
    	df::adj_load(var_joint_qd, var_46, adj_joint_qd, adj_46, adj_47);
    	df::adj_add(var_dof_start, var_33, adj_dof_start, adj_33, adj_46);
    	df::adj_load(var_joint_qd, var_44, adj_joint_qd, adj_44, adj_45);
    	df::adj_add(var_dof_start, var_2, adj_dof_start, adj_2, adj_44);
    	df::adj_load(var_joint_qd, var_42, adj_joint_qd, adj_42, adj_43);
    	df::adj_add(var_dof_start, var_0, adj_dof_start, adj_0, adj_42);
    	df::adj_float3(var_36, var_38, var_40, adj_36, adj_38, adj_40, adj_41);
    	df::adj_load(var_joint_q, var_39, adj_joint_q, adj_39, adj_40);
    	df::adj_add(var_coord_start, var_33, adj_coord_start, adj_33, adj_39);
    	df::adj_load(var_joint_q, var_37, adj_joint_q, adj_37, adj_38);
    	df::adj_add(var_coord_start, var_2, adj_coord_start, adj_2, adj_37);
    	df::adj_load(var_joint_q, var_35, adj_joint_q, adj_35, adj_36);
    	df::adj_add(var_coord_start, var_0, adj_coord_start, adj_0, adj_35);
    }
    if (var_4) {
    	df::adj_store(var_tau, var_dof_start, var_32, adj_tau, adj_dof_start, adj_32);
    	df::adj_add(var_31, var_22, adj_31, adj_22, adj_32);
    	df::adj_add(var_30, var_20, adj_30, adj_20, adj_31);
    	df::adj_add(var_29, var_8, adj_29, adj_8, adj_30);
    	df::adj_sub(var_27, var_28, adj_27, adj_28, adj_29);
    	df::adj_mul(var_target_k_d, var_7, adj_target_k_d, adj_7, adj_28);
    	df::adj_sub(var_24, var_26, adj_24, adj_26, adj_27);
    	df::adj_mul(var_target_k_e, var_25, adj_target_k_e, adj_25, adj_26);
    	df::adj_sub(var_6, var_9, adj_6, adj_9, adj_25);
    	df::adj_sub(var_12, var_23, adj_12, adj_23, adj_24);
    	df::adj_spatial_dot(var_5, var_body_f_s, adj_5, adj_body_f_s, adj_23);
    	df::adj_mul(var_21, var_7, adj_21, adj_7, adj_22);
    	df::adj_sub(var_12, var_limit_k_d, adj_12, adj_limit_k_d, adj_21);
    	df::adj_select(var_17, var_16, var_19, adj_17, adj_16, adj_19, adj_20);
    	if (var_17) {
    		df::adj_mul(var_limit_k_e, var_18, adj_limit_k_e, adj_18, adj_19);
    		df::adj_sub(var_11, var_6, adj_11, adj_6, adj_18);
    	}
    	df::adj_select(var_13, var_12, var_15, adj_13, adj_12, adj_15, adj_16);
    	if (var_13) {
    		df::adj_mul(var_limit_k_e, var_14, adj_limit_k_e, adj_14, adj_15);
    		df::adj_sub(var_10, var_6, adj_10, adj_6, adj_14);
    	}
    	df::adj_load(var_joint_limit_upper, var_coord_start, adj_joint_limit_upper, adj_coord_start, adj_11);
    	df::adj_load(var_joint_limit_lower, var_coord_start, adj_joint_limit_lower, adj_coord_start, adj_10);
    	df::adj_load(var_joint_target, var_coord_start, adj_joint_target, adj_coord_start, adj_9);
    	df::adj_load(var_joint_act, var_dof_start, adj_joint_act, adj_dof_start, adj_8);
    	df::adj_load(var_joint_qd, var_dof_start, adj_joint_qd, adj_dof_start, adj_7);
    	df::adj_load(var_joint_q, var_coord_start, adj_joint_q, adj_coord_start, adj_6);
    	df::adj_load(var_joint_S_s, var_dof_start, adj_joint_S_s, adj_dof_start, adj_5);
    }
    return;

}


int jcalc_integrate_cpu_func(
	int var_type,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_qdd,
	int var_coord_start,
	int var_dof_start,
	float var_dt,
	float* var_joint_q_new,
	float* var_joint_qd_new)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    const int var_2 = 1;
    bool var_3;
    bool var_4;
    float var_5;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    float var_10;
    float var_11;
    const int var_12 = 2;
    bool var_13;
    int var_14;
    float var_15;
    int var_16;
    float var_17;
    int var_18;
    float var_19;
    df::float3 var_20;
    int var_21;
    float var_22;
    int var_23;
    float var_24;
    int var_25;
    float var_26;
    df::float3 var_27;
    int var_28;
    float var_29;
    int var_30;
    float var_31;
    int var_32;
    float var_33;
    const int var_34 = 3;
    int var_35;
    float var_36;
    quat var_37;
    df::float3 var_38;
    df::float3 var_39;
    const float var_40 = 0.0;
    quat var_41;
    quat var_42;
    const float var_43 = 0.5;
    quat var_44;
    quat var_45;
    quat var_46;
    quat var_47;
    int var_48;
    float var_49;
    int var_50;
    float var_51;
    int var_52;
    float var_53;
    int var_54;
    float var_55;
    int var_56;
    float var_57;
    int var_58;
    float var_59;
    int var_60;
    float var_61;
    const int var_62 = 4;
    bool var_63;
    int var_64;
    float var_65;
    int var_66;
    float var_67;
    int var_68;
    float var_69;
    df::float3 var_70;
    int var_71;
    float var_72;
    int var_73;
    float var_74;
    const int var_75 = 5;
    int var_76;
    float var_77;
    df::float3 var_78;
    int var_79;
    float var_80;
    int var_81;
    float var_82;
    int var_83;
    float var_84;
    df::float3 var_85;
    int var_86;
    float var_87;
    int var_88;
    float var_89;
    int var_90;
    float var_91;
    df::float3 var_92;
    df::float3 var_93;
    df::float3 var_94;
    df::float3 var_95;
    df::float3 var_96;
    int var_97;
    float var_98;
    int var_99;
    float var_100;
    int var_101;
    float var_102;
    df::float3 var_103;
    df::float3 var_104;
    df::float3 var_105;
    int var_106;
    float var_107;
    int var_108;
    float var_109;
    int var_110;
    float var_111;
    const int var_112 = 6;
    int var_113;
    float var_114;
    quat var_115;
    quat var_116;
    quat var_117;
    quat var_118;
    df::float3 var_119;
    df::float3 var_120;
    quat var_121;
    quat var_122;
    quat var_123;
    int var_124;
    float var_125;
    int var_126;
    float var_127;
    int var_128;
    float var_129;
    int var_130;
    float var_131;
    int var_132;
    float var_133;
    int var_134;
    float var_135;
    int var_136;
    float var_137;
    int var_138;
    float var_139;
    int var_140;
    float var_141;
    int var_142;
    float var_143;
    int var_144;
    float var_145;
    int var_146;
    float var_147;
    int var_148;
    float var_149;
    //---------
    // forward
    var_1 = (var_type == var_0);
    var_3 = (var_type == var_2);
    var_4 = var_1 || var_3;
    if (var_4) {
    	var_5 = df::load(var_joint_qdd, var_dof_start);
    	var_6 = df::load(var_joint_qd, var_dof_start);
    	var_7 = df::load(var_joint_q, var_coord_start);
    	var_8 = df::mul(var_5, var_dt);
    	var_9 = df::add(var_6, var_8);
    	var_10 = df::mul(var_9, var_dt);
    	var_11 = df::add(var_7, var_10);
    	df::store(var_joint_qd_new, var_dof_start, var_9);
    	df::store(var_joint_q_new, var_coord_start, var_11);
    }
    var_13 = (var_type == var_12);
    if (var_13) {
    	var_14 = df::add(var_dof_start, var_0);
    	var_15 = df::load(var_joint_qdd, var_14);
    	var_16 = df::add(var_dof_start, var_2);
    	var_17 = df::load(var_joint_qdd, var_16);
    	var_18 = df::add(var_dof_start, var_12);
    	var_19 = df::load(var_joint_qdd, var_18);
    	var_20 = df::float3(var_15, var_17, var_19);
    	var_21 = df::add(var_dof_start, var_0);
    	var_22 = df::load(var_joint_qd, var_21);
    	var_23 = df::add(var_dof_start, var_2);
    	var_24 = df::load(var_joint_qd, var_23);
    	var_25 = df::add(var_dof_start, var_12);
    	var_26 = df::load(var_joint_qd, var_25);
    	var_27 = df::float3(var_22, var_24, var_26);
    	var_28 = df::add(var_coord_start, var_0);
    	var_29 = df::load(var_joint_q, var_28);
    	var_30 = df::add(var_coord_start, var_2);
    	var_31 = df::load(var_joint_q, var_30);
    	var_32 = df::add(var_coord_start, var_12);
    	var_33 = df::load(var_joint_q, var_32);
    	var_35 = df::add(var_coord_start, var_34);
    	var_36 = df::load(var_joint_q, var_35);
    	var_37 = df::quat(var_29, var_31, var_33, var_36);
    	var_38 = df::mul(var_20, var_dt);
    	var_39 = df::add(var_27, var_38);
    	var_41 = df::quat(var_39, var_40);
    	var_42 = df::mul(var_41, var_37);
    	var_44 = df::mul(var_42, var_43);
    	var_45 = df::mul(var_44, var_dt);
    	var_46 = df::add(var_37, var_45);
    	var_47 = df::normalize(var_46);
    	var_48 = df::add(var_coord_start, var_0);
    	var_49 = df::index(var_47, var_0);
    	df::store(var_joint_q_new, var_48, var_49);
    	var_50 = df::add(var_coord_start, var_2);
    	var_51 = df::index(var_47, var_2);
    	df::store(var_joint_q_new, var_50, var_51);
    	var_52 = df::add(var_coord_start, var_12);
    	var_53 = df::index(var_47, var_12);
    	df::store(var_joint_q_new, var_52, var_53);
    	var_54 = df::add(var_coord_start, var_34);
    	var_55 = df::index(var_47, var_34);
    	df::store(var_joint_q_new, var_54, var_55);
    	var_56 = df::add(var_dof_start, var_0);
    	var_57 = df::index(var_39, var_0);
    	df::store(var_joint_qd_new, var_56, var_57);
    	var_58 = df::add(var_dof_start, var_2);
    	var_59 = df::index(var_39, var_2);
    	df::store(var_joint_qd_new, var_58, var_59);
    	var_60 = df::add(var_dof_start, var_12);
    	var_61 = df::index(var_39, var_12);
    	df::store(var_joint_qd_new, var_60, var_61);
    }
    var_63 = (var_type == var_62);
    if (var_63) {
    	var_64 = df::add(var_dof_start, var_0);
    	var_65 = df::load(var_joint_qdd, var_64);
    	var_66 = df::add(var_dof_start, var_2);
    	var_67 = df::load(var_joint_qdd, var_66);
    	var_68 = df::add(var_dof_start, var_12);
    	var_69 = df::load(var_joint_qdd, var_68);
    	var_70 = df::float3(var_65, var_67, var_69);
    	var_71 = df::add(var_dof_start, var_34);
    	var_72 = df::load(var_joint_qdd, var_71);
    	var_73 = df::add(var_dof_start, var_62);
    	var_74 = df::load(var_joint_qdd, var_73);
    	var_76 = df::add(var_dof_start, var_75);
    	var_77 = df::load(var_joint_qdd, var_76);
    	var_78 = df::float3(var_72, var_74, var_77);
    	var_79 = df::add(var_dof_start, var_0);
    	var_80 = df::load(var_joint_qd, var_79);
    	var_81 = df::add(var_dof_start, var_2);
    	var_82 = df::load(var_joint_qd, var_81);
    	var_83 = df::add(var_dof_start, var_12);
    	var_84 = df::load(var_joint_qd, var_83);
    	var_85 = df::float3(var_80, var_82, var_84);
    	var_86 = df::add(var_dof_start, var_34);
    	var_87 = df::load(var_joint_qd, var_86);
    	var_88 = df::add(var_dof_start, var_62);
    	var_89 = df::load(var_joint_qd, var_88);
    	var_90 = df::add(var_dof_start, var_75);
    	var_91 = df::load(var_joint_qd, var_90);
    	var_92 = df::float3(var_87, var_89, var_91);
    	var_93 = df::mul(var_70, var_dt);
    	var_94 = df::add(var_85, var_93);
    	var_95 = df::mul(var_78, var_dt);
    	var_96 = df::add(var_92, var_95);
    	var_97 = df::add(var_coord_start, var_0);
    	var_98 = df::load(var_joint_q, var_97);
    	var_99 = df::add(var_coord_start, var_2);
    	var_100 = df::load(var_joint_q, var_99);
    	var_101 = df::add(var_coord_start, var_12);
    	var_102 = df::load(var_joint_q, var_101);
    	var_103 = df::float3(var_98, var_100, var_102);
    	var_104 = df::cross(var_94, var_103);
    	var_105 = df::add(var_96, var_104);
    	var_106 = df::add(var_coord_start, var_34);
    	var_107 = df::load(var_joint_q, var_106);
    	var_108 = df::add(var_coord_start, var_62);
    	var_109 = df::load(var_joint_q, var_108);
    	var_110 = df::add(var_coord_start, var_75);
    	var_111 = df::load(var_joint_q, var_110);
    	var_113 = df::add(var_coord_start, var_112);
    	var_114 = df::load(var_joint_q, var_113);
    	var_115 = df::quat(var_107, var_109, var_111, var_114);
    	var_116 = df::quat(var_94, var_40);
    	var_117 = df::mul(var_116, var_115);
    	var_118 = df::mul(var_117, var_43);
    	var_119 = df::mul(var_105, var_dt);
    	var_120 = df::add(var_103, var_119);
    	var_121 = df::mul(var_118, var_dt);
    	var_122 = df::add(var_115, var_121);
    	var_123 = df::normalize(var_122);
    	var_124 = df::add(var_coord_start, var_0);
    	var_125 = df::index(var_120, var_0);
    	df::store(var_joint_q_new, var_124, var_125);
    	var_126 = df::add(var_coord_start, var_2);
    	var_127 = df::index(var_120, var_2);
    	df::store(var_joint_q_new, var_126, var_127);
    	var_128 = df::add(var_coord_start, var_12);
    	var_129 = df::index(var_120, var_12);
    	df::store(var_joint_q_new, var_128, var_129);
    	var_130 = df::add(var_coord_start, var_34);
    	var_131 = df::index(var_123, var_0);
    	df::store(var_joint_q_new, var_130, var_131);
    	var_132 = df::add(var_coord_start, var_62);
    	var_133 = df::index(var_123, var_2);
    	df::store(var_joint_q_new, var_132, var_133);
    	var_134 = df::add(var_coord_start, var_75);
    	var_135 = df::index(var_123, var_12);
    	df::store(var_joint_q_new, var_134, var_135);
    	var_136 = df::add(var_coord_start, var_112);
    	var_137 = df::index(var_123, var_34);
    	df::store(var_joint_q_new, var_136, var_137);
    	var_138 = df::add(var_dof_start, var_0);
    	var_139 = df::index(var_94, var_0);
    	df::store(var_joint_qd_new, var_138, var_139);
    	var_140 = df::add(var_dof_start, var_2);
    	var_141 = df::index(var_94, var_2);
    	df::store(var_joint_qd_new, var_140, var_141);
    	var_142 = df::add(var_dof_start, var_12);
    	var_143 = df::index(var_94, var_12);
    	df::store(var_joint_qd_new, var_142, var_143);
    	var_144 = df::add(var_dof_start, var_34);
    	var_145 = df::index(var_96, var_0);
    	df::store(var_joint_qd_new, var_144, var_145);
    	var_146 = df::add(var_dof_start, var_62);
    	var_147 = df::index(var_96, var_2);
    	df::store(var_joint_qd_new, var_146, var_147);
    	var_148 = df::add(var_dof_start, var_75);
    	var_149 = df::index(var_96, var_12);
    	df::store(var_joint_qd_new, var_148, var_149);
    }
    return var_0;
    

}

void adj_jcalc_integrate_cpu_func(
	int var_type,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_qdd,
	int var_coord_start,
	int var_dof_start,
	float var_dt,
	float* var_joint_q_new,
	float* var_joint_qd_new, 
	int & adj_type,
	float* adj_joint_q,
	float* adj_joint_qd,
	float* adj_joint_qdd,
	int & adj_coord_start,
	int & adj_dof_start,
	float & adj_dt,
	float* adj_joint_q_new,
	float* adj_joint_qd_new,
	int & adj_ret)
{
        //---------
    // primal vars
    const int var_0 = 0;
    bool var_1;
    const int var_2 = 1;
    bool var_3;
    bool var_4;
    float var_5;
    float var_6;
    float var_7;
    float var_8;
    float var_9;
    float var_10;
    float var_11;
    const int var_12 = 2;
    bool var_13;
    int var_14;
    float var_15;
    int var_16;
    float var_17;
    int var_18;
    float var_19;
    df::float3 var_20;
    int var_21;
    float var_22;
    int var_23;
    float var_24;
    int var_25;
    float var_26;
    df::float3 var_27;
    int var_28;
    float var_29;
    int var_30;
    float var_31;
    int var_32;
    float var_33;
    const int var_34 = 3;
    int var_35;
    float var_36;
    quat var_37;
    df::float3 var_38;
    df::float3 var_39;
    const float var_40 = 0.0;
    quat var_41;
    quat var_42;
    const float var_43 = 0.5;
    quat var_44;
    quat var_45;
    quat var_46;
    quat var_47;
    int var_48;
    float var_49;
    int var_50;
    float var_51;
    int var_52;
    float var_53;
    int var_54;
    float var_55;
    int var_56;
    float var_57;
    int var_58;
    float var_59;
    int var_60;
    float var_61;
    const int var_62 = 4;
    bool var_63;
    int var_64;
    float var_65;
    int var_66;
    float var_67;
    int var_68;
    float var_69;
    df::float3 var_70;
    int var_71;
    float var_72;
    int var_73;
    float var_74;
    const int var_75 = 5;
    int var_76;
    float var_77;
    df::float3 var_78;
    int var_79;
    float var_80;
    int var_81;
    float var_82;
    int var_83;
    float var_84;
    df::float3 var_85;
    int var_86;
    float var_87;
    int var_88;
    float var_89;
    int var_90;
    float var_91;
    df::float3 var_92;
    df::float3 var_93;
    df::float3 var_94;
    df::float3 var_95;
    df::float3 var_96;
    int var_97;
    float var_98;
    int var_99;
    float var_100;
    int var_101;
    float var_102;
    df::float3 var_103;
    df::float3 var_104;
    df::float3 var_105;
    int var_106;
    float var_107;
    int var_108;
    float var_109;
    int var_110;
    float var_111;
    const int var_112 = 6;
    int var_113;
    float var_114;
    quat var_115;
    quat var_116;
    quat var_117;
    quat var_118;
    df::float3 var_119;
    df::float3 var_120;
    quat var_121;
    quat var_122;
    quat var_123;
    int var_124;
    float var_125;
    int var_126;
    float var_127;
    int var_128;
    float var_129;
    int var_130;
    float var_131;
    int var_132;
    float var_133;
    int var_134;
    float var_135;
    int var_136;
    float var_137;
    int var_138;
    float var_139;
    int var_140;
    float var_141;
    int var_142;
    float var_143;
    int var_144;
    float var_145;
    int var_146;
    float var_147;
    int var_148;
    float var_149;
    //---------
    // dual vars
    int adj_0 = 0;
    bool adj_1 = 0;
    int adj_2 = 0;
    bool adj_3 = 0;
    bool adj_4 = 0;
    float adj_5 = 0;
    float adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    float adj_10 = 0;
    float adj_11 = 0;
    int adj_12 = 0;
    bool adj_13 = 0;
    int adj_14 = 0;
    float adj_15 = 0;
    int adj_16 = 0;
    float adj_17 = 0;
    int adj_18 = 0;
    float adj_19 = 0;
    df::float3 adj_20 = 0;
    int adj_21 = 0;
    float adj_22 = 0;
    int adj_23 = 0;
    float adj_24 = 0;
    int adj_25 = 0;
    float adj_26 = 0;
    df::float3 adj_27 = 0;
    int adj_28 = 0;
    float adj_29 = 0;
    int adj_30 = 0;
    float adj_31 = 0;
    int adj_32 = 0;
    float adj_33 = 0;
    int adj_34 = 0;
    int adj_35 = 0;
    float adj_36 = 0;
    quat adj_37 = 0;
    df::float3 adj_38 = 0;
    df::float3 adj_39 = 0;
    float adj_40 = 0;
    quat adj_41 = 0;
    quat adj_42 = 0;
    float adj_43 = 0;
    quat adj_44 = 0;
    quat adj_45 = 0;
    quat adj_46 = 0;
    quat adj_47 = 0;
    int adj_48 = 0;
    float adj_49 = 0;
    int adj_50 = 0;
    float adj_51 = 0;
    int adj_52 = 0;
    float adj_53 = 0;
    int adj_54 = 0;
    float adj_55 = 0;
    int adj_56 = 0;
    float adj_57 = 0;
    int adj_58 = 0;
    float adj_59 = 0;
    int adj_60 = 0;
    float adj_61 = 0;
    int adj_62 = 0;
    bool adj_63 = 0;
    int adj_64 = 0;
    float adj_65 = 0;
    int adj_66 = 0;
    float adj_67 = 0;
    int adj_68 = 0;
    float adj_69 = 0;
    df::float3 adj_70 = 0;
    int adj_71 = 0;
    float adj_72 = 0;
    int adj_73 = 0;
    float adj_74 = 0;
    int adj_75 = 0;
    int adj_76 = 0;
    float adj_77 = 0;
    df::float3 adj_78 = 0;
    int adj_79 = 0;
    float adj_80 = 0;
    int adj_81 = 0;
    float adj_82 = 0;
    int adj_83 = 0;
    float adj_84 = 0;
    df::float3 adj_85 = 0;
    int adj_86 = 0;
    float adj_87 = 0;
    int adj_88 = 0;
    float adj_89 = 0;
    int adj_90 = 0;
    float adj_91 = 0;
    df::float3 adj_92 = 0;
    df::float3 adj_93 = 0;
    df::float3 adj_94 = 0;
    df::float3 adj_95 = 0;
    df::float3 adj_96 = 0;
    int adj_97 = 0;
    float adj_98 = 0;
    int adj_99 = 0;
    float adj_100 = 0;
    int adj_101 = 0;
    float adj_102 = 0;
    df::float3 adj_103 = 0;
    df::float3 adj_104 = 0;
    df::float3 adj_105 = 0;
    int adj_106 = 0;
    float adj_107 = 0;
    int adj_108 = 0;
    float adj_109 = 0;
    int adj_110 = 0;
    float adj_111 = 0;
    int adj_112 = 0;
    int adj_113 = 0;
    float adj_114 = 0;
    quat adj_115 = 0;
    quat adj_116 = 0;
    quat adj_117 = 0;
    quat adj_118 = 0;
    df::float3 adj_119 = 0;
    df::float3 adj_120 = 0;
    quat adj_121 = 0;
    quat adj_122 = 0;
    quat adj_123 = 0;
    int adj_124 = 0;
    float adj_125 = 0;
    int adj_126 = 0;
    float adj_127 = 0;
    int adj_128 = 0;
    float adj_129 = 0;
    int adj_130 = 0;
    float adj_131 = 0;
    int adj_132 = 0;
    float adj_133 = 0;
    int adj_134 = 0;
    float adj_135 = 0;
    int adj_136 = 0;
    float adj_137 = 0;
    int adj_138 = 0;
    float adj_139 = 0;
    int adj_140 = 0;
    float adj_141 = 0;
    int adj_142 = 0;
    float adj_143 = 0;
    int adj_144 = 0;
    float adj_145 = 0;
    int adj_146 = 0;
    float adj_147 = 0;
    int adj_148 = 0;
    float adj_149 = 0;
    //---------
    // forward
    var_1 = (var_type == var_0);
    var_3 = (var_type == var_2);
    var_4 = var_1 || var_3;
    if (var_4) {
    	var_5 = df::load(var_joint_qdd, var_dof_start);
    	var_6 = df::load(var_joint_qd, var_dof_start);
    	var_7 = df::load(var_joint_q, var_coord_start);
    	var_8 = df::mul(var_5, var_dt);
    	var_9 = df::add(var_6, var_8);
    	var_10 = df::mul(var_9, var_dt);
    	var_11 = df::add(var_7, var_10);
    	df::store(var_joint_qd_new, var_dof_start, var_9);
    	df::store(var_joint_q_new, var_coord_start, var_11);
    }
    var_13 = (var_type == var_12);
    if (var_13) {
    	var_14 = df::add(var_dof_start, var_0);
    	var_15 = df::load(var_joint_qdd, var_14);
    	var_16 = df::add(var_dof_start, var_2);
    	var_17 = df::load(var_joint_qdd, var_16);
    	var_18 = df::add(var_dof_start, var_12);
    	var_19 = df::load(var_joint_qdd, var_18);
    	var_20 = df::float3(var_15, var_17, var_19);
    	var_21 = df::add(var_dof_start, var_0);
    	var_22 = df::load(var_joint_qd, var_21);
    	var_23 = df::add(var_dof_start, var_2);
    	var_24 = df::load(var_joint_qd, var_23);
    	var_25 = df::add(var_dof_start, var_12);
    	var_26 = df::load(var_joint_qd, var_25);
    	var_27 = df::float3(var_22, var_24, var_26);
    	var_28 = df::add(var_coord_start, var_0);
    	var_29 = df::load(var_joint_q, var_28);
    	var_30 = df::add(var_coord_start, var_2);
    	var_31 = df::load(var_joint_q, var_30);
    	var_32 = df::add(var_coord_start, var_12);
    	var_33 = df::load(var_joint_q, var_32);
    	var_35 = df::add(var_coord_start, var_34);
    	var_36 = df::load(var_joint_q, var_35);
    	var_37 = df::quat(var_29, var_31, var_33, var_36);
    	var_38 = df::mul(var_20, var_dt);
    	var_39 = df::add(var_27, var_38);
    	var_41 = df::quat(var_39, var_40);
    	var_42 = df::mul(var_41, var_37);
    	var_44 = df::mul(var_42, var_43);
    	var_45 = df::mul(var_44, var_dt);
    	var_46 = df::add(var_37, var_45);
    	var_47 = df::normalize(var_46);
    	var_48 = df::add(var_coord_start, var_0);
    	var_49 = df::index(var_47, var_0);
    	df::store(var_joint_q_new, var_48, var_49);
    	var_50 = df::add(var_coord_start, var_2);
    	var_51 = df::index(var_47, var_2);
    	df::store(var_joint_q_new, var_50, var_51);
    	var_52 = df::add(var_coord_start, var_12);
    	var_53 = df::index(var_47, var_12);
    	df::store(var_joint_q_new, var_52, var_53);
    	var_54 = df::add(var_coord_start, var_34);
    	var_55 = df::index(var_47, var_34);
    	df::store(var_joint_q_new, var_54, var_55);
    	var_56 = df::add(var_dof_start, var_0);
    	var_57 = df::index(var_39, var_0);
    	df::store(var_joint_qd_new, var_56, var_57);
    	var_58 = df::add(var_dof_start, var_2);
    	var_59 = df::index(var_39, var_2);
    	df::store(var_joint_qd_new, var_58, var_59);
    	var_60 = df::add(var_dof_start, var_12);
    	var_61 = df::index(var_39, var_12);
    	df::store(var_joint_qd_new, var_60, var_61);
    }
    var_63 = (var_type == var_62);
    if (var_63) {
    	var_64 = df::add(var_dof_start, var_0);
    	var_65 = df::load(var_joint_qdd, var_64);
    	var_66 = df::add(var_dof_start, var_2);
    	var_67 = df::load(var_joint_qdd, var_66);
    	var_68 = df::add(var_dof_start, var_12);
    	var_69 = df::load(var_joint_qdd, var_68);
    	var_70 = df::float3(var_65, var_67, var_69);
    	var_71 = df::add(var_dof_start, var_34);
    	var_72 = df::load(var_joint_qdd, var_71);
    	var_73 = df::add(var_dof_start, var_62);
    	var_74 = df::load(var_joint_qdd, var_73);
    	var_76 = df::add(var_dof_start, var_75);
    	var_77 = df::load(var_joint_qdd, var_76);
    	var_78 = df::float3(var_72, var_74, var_77);
    	var_79 = df::add(var_dof_start, var_0);
    	var_80 = df::load(var_joint_qd, var_79);
    	var_81 = df::add(var_dof_start, var_2);
    	var_82 = df::load(var_joint_qd, var_81);
    	var_83 = df::add(var_dof_start, var_12);
    	var_84 = df::load(var_joint_qd, var_83);
    	var_85 = df::float3(var_80, var_82, var_84);
    	var_86 = df::add(var_dof_start, var_34);
    	var_87 = df::load(var_joint_qd, var_86);
    	var_88 = df::add(var_dof_start, var_62);
    	var_89 = df::load(var_joint_qd, var_88);
    	var_90 = df::add(var_dof_start, var_75);
    	var_91 = df::load(var_joint_qd, var_90);
    	var_92 = df::float3(var_87, var_89, var_91);
    	var_93 = df::mul(var_70, var_dt);
    	var_94 = df::add(var_85, var_93);
    	var_95 = df::mul(var_78, var_dt);
    	var_96 = df::add(var_92, var_95);
    	var_97 = df::add(var_coord_start, var_0);
    	var_98 = df::load(var_joint_q, var_97);
    	var_99 = df::add(var_coord_start, var_2);
    	var_100 = df::load(var_joint_q, var_99);
    	var_101 = df::add(var_coord_start, var_12);
    	var_102 = df::load(var_joint_q, var_101);
    	var_103 = df::float3(var_98, var_100, var_102);
    	var_104 = df::cross(var_94, var_103);
    	var_105 = df::add(var_96, var_104);
    	var_106 = df::add(var_coord_start, var_34);
    	var_107 = df::load(var_joint_q, var_106);
    	var_108 = df::add(var_coord_start, var_62);
    	var_109 = df::load(var_joint_q, var_108);
    	var_110 = df::add(var_coord_start, var_75);
    	var_111 = df::load(var_joint_q, var_110);
    	var_113 = df::add(var_coord_start, var_112);
    	var_114 = df::load(var_joint_q, var_113);
    	var_115 = df::quat(var_107, var_109, var_111, var_114);
    	var_116 = df::quat(var_94, var_40);
    	var_117 = df::mul(var_116, var_115);
    	var_118 = df::mul(var_117, var_43);
    	var_119 = df::mul(var_105, var_dt);
    	var_120 = df::add(var_103, var_119);
    	var_121 = df::mul(var_118, var_dt);
    	var_122 = df::add(var_115, var_121);
    	var_123 = df::normalize(var_122);
    	var_124 = df::add(var_coord_start, var_0);
    	var_125 = df::index(var_120, var_0);
    	df::store(var_joint_q_new, var_124, var_125);
    	var_126 = df::add(var_coord_start, var_2);
    	var_127 = df::index(var_120, var_2);
    	df::store(var_joint_q_new, var_126, var_127);
    	var_128 = df::add(var_coord_start, var_12);
    	var_129 = df::index(var_120, var_12);
    	df::store(var_joint_q_new, var_128, var_129);
    	var_130 = df::add(var_coord_start, var_34);
    	var_131 = df::index(var_123, var_0);
    	df::store(var_joint_q_new, var_130, var_131);
    	var_132 = df::add(var_coord_start, var_62);
    	var_133 = df::index(var_123, var_2);
    	df::store(var_joint_q_new, var_132, var_133);
    	var_134 = df::add(var_coord_start, var_75);
    	var_135 = df::index(var_123, var_12);
    	df::store(var_joint_q_new, var_134, var_135);
    	var_136 = df::add(var_coord_start, var_112);
    	var_137 = df::index(var_123, var_34);
    	df::store(var_joint_q_new, var_136, var_137);
    	var_138 = df::add(var_dof_start, var_0);
    	var_139 = df::index(var_94, var_0);
    	df::store(var_joint_qd_new, var_138, var_139);
    	var_140 = df::add(var_dof_start, var_2);
    	var_141 = df::index(var_94, var_2);
    	df::store(var_joint_qd_new, var_140, var_141);
    	var_142 = df::add(var_dof_start, var_12);
    	var_143 = df::index(var_94, var_12);
    	df::store(var_joint_qd_new, var_142, var_143);
    	var_144 = df::add(var_dof_start, var_34);
    	var_145 = df::index(var_96, var_0);
    	df::store(var_joint_qd_new, var_144, var_145);
    	var_146 = df::add(var_dof_start, var_62);
    	var_147 = df::index(var_96, var_2);
    	df::store(var_joint_qd_new, var_146, var_147);
    	var_148 = df::add(var_dof_start, var_75);
    	var_149 = df::index(var_96, var_12);
    	df::store(var_joint_qd_new, var_148, var_149);
    }
    goto label0;
    //---------
    // reverse
    label0:;
    adj_0 += adj_ret;
    if (var_63) {
    	df::adj_store(var_joint_qd_new, var_148, var_149, adj_joint_qd_new, adj_148, adj_149);
    	df::adj_index(var_96, var_12, adj_96, adj_12, adj_149);
    	df::adj_add(var_dof_start, var_75, adj_dof_start, adj_75, adj_148);
    	df::adj_store(var_joint_qd_new, var_146, var_147, adj_joint_qd_new, adj_146, adj_147);
    	df::adj_index(var_96, var_2, adj_96, adj_2, adj_147);
    	df::adj_add(var_dof_start, var_62, adj_dof_start, adj_62, adj_146);
    	df::adj_store(var_joint_qd_new, var_144, var_145, adj_joint_qd_new, adj_144, adj_145);
    	df::adj_index(var_96, var_0, adj_96, adj_0, adj_145);
    	df::adj_add(var_dof_start, var_34, adj_dof_start, adj_34, adj_144);
    	df::adj_store(var_joint_qd_new, var_142, var_143, adj_joint_qd_new, adj_142, adj_143);
    	df::adj_index(var_94, var_12, adj_94, adj_12, adj_143);
    	df::adj_add(var_dof_start, var_12, adj_dof_start, adj_12, adj_142);
    	df::adj_store(var_joint_qd_new, var_140, var_141, adj_joint_qd_new, adj_140, adj_141);
    	df::adj_index(var_94, var_2, adj_94, adj_2, adj_141);
    	df::adj_add(var_dof_start, var_2, adj_dof_start, adj_2, adj_140);
    	df::adj_store(var_joint_qd_new, var_138, var_139, adj_joint_qd_new, adj_138, adj_139);
    	df::adj_index(var_94, var_0, adj_94, adj_0, adj_139);
    	df::adj_add(var_dof_start, var_0, adj_dof_start, adj_0, adj_138);
    	df::adj_store(var_joint_q_new, var_136, var_137, adj_joint_q_new, adj_136, adj_137);
    	df::adj_index(var_123, var_34, adj_123, adj_34, adj_137);
    	df::adj_add(var_coord_start, var_112, adj_coord_start, adj_112, adj_136);
    	df::adj_store(var_joint_q_new, var_134, var_135, adj_joint_q_new, adj_134, adj_135);
    	df::adj_index(var_123, var_12, adj_123, adj_12, adj_135);
    	df::adj_add(var_coord_start, var_75, adj_coord_start, adj_75, adj_134);
    	df::adj_store(var_joint_q_new, var_132, var_133, adj_joint_q_new, adj_132, adj_133);
    	df::adj_index(var_123, var_2, adj_123, adj_2, adj_133);
    	df::adj_add(var_coord_start, var_62, adj_coord_start, adj_62, adj_132);
    	df::adj_store(var_joint_q_new, var_130, var_131, adj_joint_q_new, adj_130, adj_131);
    	df::adj_index(var_123, var_0, adj_123, adj_0, adj_131);
    	df::adj_add(var_coord_start, var_34, adj_coord_start, adj_34, adj_130);
    	df::adj_store(var_joint_q_new, var_128, var_129, adj_joint_q_new, adj_128, adj_129);
    	df::adj_index(var_120, var_12, adj_120, adj_12, adj_129);
    	df::adj_add(var_coord_start, var_12, adj_coord_start, adj_12, adj_128);
    	df::adj_store(var_joint_q_new, var_126, var_127, adj_joint_q_new, adj_126, adj_127);
    	df::adj_index(var_120, var_2, adj_120, adj_2, adj_127);
    	df::adj_add(var_coord_start, var_2, adj_coord_start, adj_2, adj_126);
    	df::adj_store(var_joint_q_new, var_124, var_125, adj_joint_q_new, adj_124, adj_125);
    	df::adj_index(var_120, var_0, adj_120, adj_0, adj_125);
    	df::adj_add(var_coord_start, var_0, adj_coord_start, adj_0, adj_124);
    	df::adj_normalize(var_122, adj_122, adj_123);
    	df::adj_add(var_115, var_121, adj_115, adj_121, adj_122);
    	df::adj_mul(var_118, var_dt, adj_118, adj_dt, adj_121);
    	df::adj_add(var_103, var_119, adj_103, adj_119, adj_120);
    	df::adj_mul(var_105, var_dt, adj_105, adj_dt, adj_119);
    	df::adj_mul(var_117, var_43, adj_117, adj_43, adj_118);
    	df::adj_mul(var_116, var_115, adj_116, adj_115, adj_117);
    	df::adj_quat(var_94, var_40, adj_94, adj_40, adj_116);
    	df::adj_quat(var_107, var_109, var_111, var_114, adj_107, adj_109, adj_111, adj_114, adj_115);
    	df::adj_load(var_joint_q, var_113, adj_joint_q, adj_113, adj_114);
    	df::adj_add(var_coord_start, var_112, adj_coord_start, adj_112, adj_113);
    	df::adj_load(var_joint_q, var_110, adj_joint_q, adj_110, adj_111);
    	df::adj_add(var_coord_start, var_75, adj_coord_start, adj_75, adj_110);
    	df::adj_load(var_joint_q, var_108, adj_joint_q, adj_108, adj_109);
    	df::adj_add(var_coord_start, var_62, adj_coord_start, adj_62, adj_108);
    	df::adj_load(var_joint_q, var_106, adj_joint_q, adj_106, adj_107);
    	df::adj_add(var_coord_start, var_34, adj_coord_start, adj_34, adj_106);
    	df::adj_add(var_96, var_104, adj_96, adj_104, adj_105);
    	df::adj_cross(var_94, var_103, adj_94, adj_103, adj_104);
    	df::adj_float3(var_98, var_100, var_102, adj_98, adj_100, adj_102, adj_103);
    	df::adj_load(var_joint_q, var_101, adj_joint_q, adj_101, adj_102);
    	df::adj_add(var_coord_start, var_12, adj_coord_start, adj_12, adj_101);
    	df::adj_load(var_joint_q, var_99, adj_joint_q, adj_99, adj_100);
    	df::adj_add(var_coord_start, var_2, adj_coord_start, adj_2, adj_99);
    	df::adj_load(var_joint_q, var_97, adj_joint_q, adj_97, adj_98);
    	df::adj_add(var_coord_start, var_0, adj_coord_start, adj_0, adj_97);
    	df::adj_add(var_92, var_95, adj_92, adj_95, adj_96);
    	df::adj_mul(var_78, var_dt, adj_78, adj_dt, adj_95);
    	df::adj_add(var_85, var_93, adj_85, adj_93, adj_94);
    	df::adj_mul(var_70, var_dt, adj_70, adj_dt, adj_93);
    	df::adj_float3(var_87, var_89, var_91, adj_87, adj_89, adj_91, adj_92);
    	df::adj_load(var_joint_qd, var_90, adj_joint_qd, adj_90, adj_91);
    	df::adj_add(var_dof_start, var_75, adj_dof_start, adj_75, adj_90);
    	df::adj_load(var_joint_qd, var_88, adj_joint_qd, adj_88, adj_89);
    	df::adj_add(var_dof_start, var_62, adj_dof_start, adj_62, adj_88);
    	df::adj_load(var_joint_qd, var_86, adj_joint_qd, adj_86, adj_87);
    	df::adj_add(var_dof_start, var_34, adj_dof_start, adj_34, adj_86);
    	df::adj_float3(var_80, var_82, var_84, adj_80, adj_82, adj_84, adj_85);
    	df::adj_load(var_joint_qd, var_83, adj_joint_qd, adj_83, adj_84);
    	df::adj_add(var_dof_start, var_12, adj_dof_start, adj_12, adj_83);
    	df::adj_load(var_joint_qd, var_81, adj_joint_qd, adj_81, adj_82);
    	df::adj_add(var_dof_start, var_2, adj_dof_start, adj_2, adj_81);
    	df::adj_load(var_joint_qd, var_79, adj_joint_qd, adj_79, adj_80);
    	df::adj_add(var_dof_start, var_0, adj_dof_start, adj_0, adj_79);
    	df::adj_float3(var_72, var_74, var_77, adj_72, adj_74, adj_77, adj_78);
    	df::adj_load(var_joint_qdd, var_76, adj_joint_qdd, adj_76, adj_77);
    	df::adj_add(var_dof_start, var_75, adj_dof_start, adj_75, adj_76);
    	df::adj_load(var_joint_qdd, var_73, adj_joint_qdd, adj_73, adj_74);
    	df::adj_add(var_dof_start, var_62, adj_dof_start, adj_62, adj_73);
    	df::adj_load(var_joint_qdd, var_71, adj_joint_qdd, adj_71, adj_72);
    	df::adj_add(var_dof_start, var_34, adj_dof_start, adj_34, adj_71);
    	df::adj_float3(var_65, var_67, var_69, adj_65, adj_67, adj_69, adj_70);
    	df::adj_load(var_joint_qdd, var_68, adj_joint_qdd, adj_68, adj_69);
    	df::adj_add(var_dof_start, var_12, adj_dof_start, adj_12, adj_68);
    	df::adj_load(var_joint_qdd, var_66, adj_joint_qdd, adj_66, adj_67);
    	df::adj_add(var_dof_start, var_2, adj_dof_start, adj_2, adj_66);
    	df::adj_load(var_joint_qdd, var_64, adj_joint_qdd, adj_64, adj_65);
    	df::adj_add(var_dof_start, var_0, adj_dof_start, adj_0, adj_64);
    }
    if (var_13) {
    	df::adj_store(var_joint_qd_new, var_60, var_61, adj_joint_qd_new, adj_60, adj_61);
    	df::adj_index(var_39, var_12, adj_39, adj_12, adj_61);
    	df::adj_add(var_dof_start, var_12, adj_dof_start, adj_12, adj_60);
    	df::adj_store(var_joint_qd_new, var_58, var_59, adj_joint_qd_new, adj_58, adj_59);
    	df::adj_index(var_39, var_2, adj_39, adj_2, adj_59);
    	df::adj_add(var_dof_start, var_2, adj_dof_start, adj_2, adj_58);
    	df::adj_store(var_joint_qd_new, var_56, var_57, adj_joint_qd_new, adj_56, adj_57);
    	df::adj_index(var_39, var_0, adj_39, adj_0, adj_57);
    	df::adj_add(var_dof_start, var_0, adj_dof_start, adj_0, adj_56);
    	df::adj_store(var_joint_q_new, var_54, var_55, adj_joint_q_new, adj_54, adj_55);
    	df::adj_index(var_47, var_34, adj_47, adj_34, adj_55);
    	df::adj_add(var_coord_start, var_34, adj_coord_start, adj_34, adj_54);
    	df::adj_store(var_joint_q_new, var_52, var_53, adj_joint_q_new, adj_52, adj_53);
    	df::adj_index(var_47, var_12, adj_47, adj_12, adj_53);
    	df::adj_add(var_coord_start, var_12, adj_coord_start, adj_12, adj_52);
    	df::adj_store(var_joint_q_new, var_50, var_51, adj_joint_q_new, adj_50, adj_51);
    	df::adj_index(var_47, var_2, adj_47, adj_2, adj_51);
    	df::adj_add(var_coord_start, var_2, adj_coord_start, adj_2, adj_50);
    	df::adj_store(var_joint_q_new, var_48, var_49, adj_joint_q_new, adj_48, adj_49);
    	df::adj_index(var_47, var_0, adj_47, adj_0, adj_49);
    	df::adj_add(var_coord_start, var_0, adj_coord_start, adj_0, adj_48);
    	df::adj_normalize(var_46, adj_46, adj_47);
    	df::adj_add(var_37, var_45, adj_37, adj_45, adj_46);
    	df::adj_mul(var_44, var_dt, adj_44, adj_dt, adj_45);
    	df::adj_mul(var_42, var_43, adj_42, adj_43, adj_44);
    	df::adj_mul(var_41, var_37, adj_41, adj_37, adj_42);
    	df::adj_quat(var_39, var_40, adj_39, adj_40, adj_41);
    	df::adj_add(var_27, var_38, adj_27, adj_38, adj_39);
    	df::adj_mul(var_20, var_dt, adj_20, adj_dt, adj_38);
    	df::adj_quat(var_29, var_31, var_33, var_36, adj_29, adj_31, adj_33, adj_36, adj_37);
    	df::adj_load(var_joint_q, var_35, adj_joint_q, adj_35, adj_36);
    	df::adj_add(var_coord_start, var_34, adj_coord_start, adj_34, adj_35);
    	df::adj_load(var_joint_q, var_32, adj_joint_q, adj_32, adj_33);
    	df::adj_add(var_coord_start, var_12, adj_coord_start, adj_12, adj_32);
    	df::adj_load(var_joint_q, var_30, adj_joint_q, adj_30, adj_31);
    	df::adj_add(var_coord_start, var_2, adj_coord_start, adj_2, adj_30);
    	df::adj_load(var_joint_q, var_28, adj_joint_q, adj_28, adj_29);
    	df::adj_add(var_coord_start, var_0, adj_coord_start, adj_0, adj_28);
    	df::adj_float3(var_22, var_24, var_26, adj_22, adj_24, adj_26, adj_27);
    	df::adj_load(var_joint_qd, var_25, adj_joint_qd, adj_25, adj_26);
    	df::adj_add(var_dof_start, var_12, adj_dof_start, adj_12, adj_25);
    	df::adj_load(var_joint_qd, var_23, adj_joint_qd, adj_23, adj_24);
    	df::adj_add(var_dof_start, var_2, adj_dof_start, adj_2, adj_23);
    	df::adj_load(var_joint_qd, var_21, adj_joint_qd, adj_21, adj_22);
    	df::adj_add(var_dof_start, var_0, adj_dof_start, adj_0, adj_21);
    	df::adj_float3(var_15, var_17, var_19, adj_15, adj_17, adj_19, adj_20);
    	df::adj_load(var_joint_qdd, var_18, adj_joint_qdd, adj_18, adj_19);
    	df::adj_add(var_dof_start, var_12, adj_dof_start, adj_12, adj_18);
    	df::adj_load(var_joint_qdd, var_16, adj_joint_qdd, adj_16, adj_17);
    	df::adj_add(var_dof_start, var_2, adj_dof_start, adj_2, adj_16);
    	df::adj_load(var_joint_qdd, var_14, adj_joint_qdd, adj_14, adj_15);
    	df::adj_add(var_dof_start, var_0, adj_dof_start, adj_0, adj_14);
    }
    if (var_4) {
    	df::adj_store(var_joint_q_new, var_coord_start, var_11, adj_joint_q_new, adj_coord_start, adj_11);
    	df::adj_store(var_joint_qd_new, var_dof_start, var_9, adj_joint_qd_new, adj_dof_start, adj_9);
    	df::adj_add(var_7, var_10, adj_7, adj_10, adj_11);
    	df::adj_mul(var_9, var_dt, adj_9, adj_dt, adj_10);
    	df::adj_add(var_6, var_8, adj_6, adj_8, adj_9);
    	df::adj_mul(var_5, var_dt, adj_5, adj_dt, adj_8);
    	df::adj_load(var_joint_q, var_coord_start, adj_joint_q, adj_coord_start, adj_7);
    	df::adj_load(var_joint_qd, var_dof_start, adj_joint_qd, adj_dof_start, adj_6);
    	df::adj_load(var_joint_qdd, var_dof_start, adj_joint_qdd, adj_dof_start, adj_5);
    }
    return;

}


int compute_link_transform_cpu_func(
	int var_i,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	spatial_transform* var_joint_X_pj,
	spatial_transform* var_joint_X_cm,
	df::float3* var_joint_axis,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm)
{
        //---------
    // primal vars
    int var_0;
    spatial_transform var_1;
    const int var_2 = 0;
    bool var_3;
    spatial_transform var_4;
    spatial_transform var_5;
    int var_6;
    df::float3 var_7;
    int var_8;
    int var_9;
    spatial_transform var_10;
    spatial_transform var_11;
    spatial_transform var_12;
    spatial_transform var_13;
    spatial_transform var_14;
    spatial_transform var_15;
    //---------
    // forward
    var_0 = df::load(var_joint_parent, var_i);
    var_1 = df::spatial_transform_identity();
    var_3 = (var_0 >= var_2);
    if (var_3) {
    	var_4 = df::load(var_body_X_sc, var_0);
    }
    var_5 = df::select(var_3, var_1, var_4);
    var_6 = df::load(var_joint_type, var_i);
    var_7 = df::load(var_joint_axis, var_i);
    var_8 = df::load(var_joint_q_start, var_i);
    var_9 = df::load(var_joint_qd_start, var_i);
    var_10 = jcalc_transform_cpu_func(var_6, var_7, var_joint_q, var_8);
    var_11 = df::load(var_joint_X_pj, var_i);
    var_12 = df::spatial_transform_multiply(var_11, var_10);
    var_13 = df::spatial_transform_multiply(var_5, var_12);
    var_14 = df::load(var_joint_X_cm, var_i);
    var_15 = df::spatial_transform_multiply(var_13, var_14);
    df::store(var_body_X_sc, var_i, var_13);
    df::store(var_body_X_sm, var_i, var_15);
    return var_2;
    

}

void adj_compute_link_transform_cpu_func(
	int var_i,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	spatial_transform* var_joint_X_pj,
	spatial_transform* var_joint_X_cm,
	df::float3* var_joint_axis,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm, 
	int & adj_i,
	int* adj_joint_type,
	int* adj_joint_parent,
	int* adj_joint_q_start,
	int* adj_joint_qd_start,
	float* adj_joint_q,
	spatial_transform* adj_joint_X_pj,
	spatial_transform* adj_joint_X_cm,
	df::float3* adj_joint_axis,
	spatial_transform* adj_body_X_sc,
	spatial_transform* adj_body_X_sm,
	int & adj_ret)
{
        //---------
    // primal vars
    int var_0;
    spatial_transform var_1;
    const int var_2 = 0;
    bool var_3;
    spatial_transform var_4;
    spatial_transform var_5;
    int var_6;
    df::float3 var_7;
    int var_8;
    int var_9;
    spatial_transform var_10;
    spatial_transform var_11;
    spatial_transform var_12;
    spatial_transform var_13;
    spatial_transform var_14;
    spatial_transform var_15;
    //---------
    // dual vars
    int adj_0 = 0;
    spatial_transform adj_1 = 0;
    int adj_2 = 0;
    bool adj_3 = 0;
    spatial_transform adj_4 = 0;
    spatial_transform adj_5 = 0;
    int adj_6 = 0;
    df::float3 adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    spatial_transform adj_10 = 0;
    spatial_transform adj_11 = 0;
    spatial_transform adj_12 = 0;
    spatial_transform adj_13 = 0;
    spatial_transform adj_14 = 0;
    spatial_transform adj_15 = 0;
    //---------
    // forward
    var_0 = df::load(var_joint_parent, var_i);
    var_1 = df::spatial_transform_identity();
    var_3 = (var_0 >= var_2);
    if (var_3) {
    	var_4 = df::load(var_body_X_sc, var_0);
    }
    var_5 = df::select(var_3, var_1, var_4);
    var_6 = df::load(var_joint_type, var_i);
    var_7 = df::load(var_joint_axis, var_i);
    var_8 = df::load(var_joint_q_start, var_i);
    var_9 = df::load(var_joint_qd_start, var_i);
    var_10 = jcalc_transform_cpu_func(var_6, var_7, var_joint_q, var_8);
    var_11 = df::load(var_joint_X_pj, var_i);
    var_12 = df::spatial_transform_multiply(var_11, var_10);
    var_13 = df::spatial_transform_multiply(var_5, var_12);
    var_14 = df::load(var_joint_X_cm, var_i);
    var_15 = df::spatial_transform_multiply(var_13, var_14);
    df::store(var_body_X_sc, var_i, var_13);
    df::store(var_body_X_sm, var_i, var_15);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_2 += adj_ret;
    df::adj_store(var_body_X_sm, var_i, var_15, adj_body_X_sm, adj_i, adj_15);
    df::adj_store(var_body_X_sc, var_i, var_13, adj_body_X_sc, adj_i, adj_13);
    df::adj_spatial_transform_multiply(var_13, var_14, adj_13, adj_14, adj_15);
    df::adj_load(var_joint_X_cm, var_i, adj_joint_X_cm, adj_i, adj_14);
    df::adj_spatial_transform_multiply(var_5, var_12, adj_5, adj_12, adj_13);
    df::adj_spatial_transform_multiply(var_11, var_10, adj_11, adj_10, adj_12);
    df::adj_load(var_joint_X_pj, var_i, adj_joint_X_pj, adj_i, adj_11);
    adj_jcalc_transform_cpu_func(var_6, var_7, var_joint_q, var_8, adj_6, adj_7, adj_joint_q, adj_8, adj_10);
    df::adj_load(var_joint_qd_start, var_i, adj_joint_qd_start, adj_i, adj_9);
    df::adj_load(var_joint_q_start, var_i, adj_joint_q_start, adj_i, adj_8);
    df::adj_load(var_joint_axis, var_i, adj_joint_axis, adj_i, adj_7);
    df::adj_load(var_joint_type, var_i, adj_joint_type, adj_i, adj_6);
    df::adj_select(var_3, var_1, var_4, adj_3, adj_1, adj_4, adj_5);
    if (var_3) {
    	df::adj_load(var_body_X_sc, var_0, adj_body_X_sc, adj_0, adj_4);
    }
    df::adj_load(var_joint_parent, var_i, adj_joint_parent, adj_i, adj_0);
    return;

}


int compute_link_velocity_cpu_func(
	int var_i,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_qd_start,
	float* var_joint_qd,
	df::float3* var_joint_axis,
	spatial_matrix* var_body_I_m,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm,
	spatial_transform* var_joint_X_pj,
	df::float3* var_gravity,
	spatial_vector* var_joint_S_s,
	spatial_matrix* var_body_I_s,
	spatial_vector* var_body_v_s,
	spatial_vector* var_body_f_s,
	spatial_vector* var_body_a_s)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    int var_2;
    int var_3;
    spatial_transform var_4;
    spatial_transform var_5;
    const int var_6 = 0;
    bool var_7;
    spatial_transform var_8;
    spatial_transform var_9;
    spatial_transform var_10;
    spatial_transform var_11;
    spatial_vector var_12;
    spatial_vector var_13;
    spatial_vector var_14;
    bool var_15;
    spatial_vector var_16;
    spatial_vector var_17;
    spatial_vector var_18;
    spatial_vector var_19;
    spatial_vector var_20;
    spatial_vector var_21;
    spatial_vector var_22;
    spatial_transform var_23;
    spatial_matrix var_24;
    df::float3 var_25;
    const int var_26 = 3;
    float var_27;
    df::float3 var_28;
    spatial_vector var_29;
    spatial_vector var_30;
    df::float3 var_31;
    quat var_32;
    spatial_transform var_33;
    spatial_vector var_34;
    spatial_matrix var_35;
    spatial_vector var_36;
    spatial_vector var_37;
    spatial_vector var_38;
    spatial_vector var_39;
    spatial_vector var_40;
    //---------
    // forward
    var_0 = df::load(var_joint_type, var_i);
    var_1 = df::load(var_joint_axis, var_i);
    var_2 = df::load(var_joint_parent, var_i);
    var_3 = df::load(var_joint_qd_start, var_i);
    var_4 = df::load(var_body_X_sc, var_i);
    var_5 = df::spatial_transform_identity();
    var_7 = (var_2 >= var_6);
    if (var_7) {
    	var_8 = df::load(var_body_X_sc, var_2);
    }
    var_9 = df::select(var_7, var_5, var_8);
    var_10 = df::load(var_joint_X_pj, var_i);
    var_11 = df::spatial_transform_multiply(var_9, var_10);
    var_12 = jcalc_motion_cpu_func(var_0, var_1, var_11, var_joint_S_s, var_joint_qd, var_3);
    var_13 = df::spatial_vector();
    var_14 = df::spatial_vector();
    var_15 = (var_2 >= var_6);
    if (var_15) {
    	var_16 = df::load(var_body_v_s, var_2);
    	var_17 = df::load(var_body_a_s, var_2);
    }
    var_18 = df::select(var_15, var_13, var_16);
    var_19 = df::select(var_15, var_14, var_17);
    var_20 = df::add(var_18, var_12);
    var_21 = df::spatial_cross(var_20, var_12);
    var_22 = df::add(var_19, var_21);
    var_23 = df::load(var_body_X_sm, var_i);
    var_24 = df::load(var_body_I_m, var_i);
    var_25 = df::load(var_gravity, var_6);
    var_27 = df::index(var_24, var_26, var_26);
    var_28 = df::float3();
    var_29 = df::spatial_vector(var_28, var_25);
    var_30 = df::mul(var_29, var_27);
    var_31 = df::spatial_transform_get_translation(var_23);
    var_32 = df::quat_identity();
    var_33 = df::spatial_transform(var_31, var_32);
    var_34 = spatial_transform_wrench_cpu_func(var_33, var_30);
    var_35 = spatial_transform_inertia_cpu_func(var_23, var_24);
    var_36 = df::mul(var_35, var_22);
    var_37 = df::mul(var_35, var_20);
    var_38 = df::spatial_cross_dual(var_20, var_37);
    var_39 = df::add(var_36, var_38);
    df::store(var_body_v_s, var_i, var_20);
    df::store(var_body_a_s, var_i, var_22);
    var_40 = df::sub(var_39, var_34);
    df::store(var_body_f_s, var_i, var_40);
    df::store(var_body_I_s, var_i, var_35);
    return var_6;
    

}

void adj_compute_link_velocity_cpu_func(
	int var_i,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_qd_start,
	float* var_joint_qd,
	df::float3* var_joint_axis,
	spatial_matrix* var_body_I_m,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm,
	spatial_transform* var_joint_X_pj,
	df::float3* var_gravity,
	spatial_vector* var_joint_S_s,
	spatial_matrix* var_body_I_s,
	spatial_vector* var_body_v_s,
	spatial_vector* var_body_f_s,
	spatial_vector* var_body_a_s, 
	int & adj_i,
	int* adj_joint_type,
	int* adj_joint_parent,
	int* adj_joint_qd_start,
	float* adj_joint_qd,
	df::float3* adj_joint_axis,
	spatial_matrix* adj_body_I_m,
	spatial_transform* adj_body_X_sc,
	spatial_transform* adj_body_X_sm,
	spatial_transform* adj_joint_X_pj,
	df::float3* adj_gravity,
	spatial_vector* adj_joint_S_s,
	spatial_matrix* adj_body_I_s,
	spatial_vector* adj_body_v_s,
	spatial_vector* adj_body_f_s,
	spatial_vector* adj_body_a_s,
	int & adj_ret)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    int var_2;
    int var_3;
    spatial_transform var_4;
    spatial_transform var_5;
    const int var_6 = 0;
    bool var_7;
    spatial_transform var_8;
    spatial_transform var_9;
    spatial_transform var_10;
    spatial_transform var_11;
    spatial_vector var_12;
    spatial_vector var_13;
    spatial_vector var_14;
    bool var_15;
    spatial_vector var_16;
    spatial_vector var_17;
    spatial_vector var_18;
    spatial_vector var_19;
    spatial_vector var_20;
    spatial_vector var_21;
    spatial_vector var_22;
    spatial_transform var_23;
    spatial_matrix var_24;
    df::float3 var_25;
    const int var_26 = 3;
    float var_27;
    df::float3 var_28;
    spatial_vector var_29;
    spatial_vector var_30;
    df::float3 var_31;
    quat var_32;
    spatial_transform var_33;
    spatial_vector var_34;
    spatial_matrix var_35;
    spatial_vector var_36;
    spatial_vector var_37;
    spatial_vector var_38;
    spatial_vector var_39;
    spatial_vector var_40;
    //---------
    // dual vars
    int adj_0 = 0;
    df::float3 adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    spatial_transform adj_4 = 0;
    spatial_transform adj_5 = 0;
    int adj_6 = 0;
    bool adj_7 = 0;
    spatial_transform adj_8 = 0;
    spatial_transform adj_9 = 0;
    spatial_transform adj_10 = 0;
    spatial_transform adj_11 = 0;
    spatial_vector adj_12 = 0;
    spatial_vector adj_13 = 0;
    spatial_vector adj_14 = 0;
    bool adj_15 = 0;
    spatial_vector adj_16 = 0;
    spatial_vector adj_17 = 0;
    spatial_vector adj_18 = 0;
    spatial_vector adj_19 = 0;
    spatial_vector adj_20 = 0;
    spatial_vector adj_21 = 0;
    spatial_vector adj_22 = 0;
    spatial_transform adj_23 = 0;
    spatial_matrix adj_24 = 0;
    df::float3 adj_25 = 0;
    int adj_26 = 0;
    float adj_27 = 0;
    df::float3 adj_28 = 0;
    spatial_vector adj_29 = 0;
    spatial_vector adj_30 = 0;
    df::float3 adj_31 = 0;
    quat adj_32 = 0;
    spatial_transform adj_33 = 0;
    spatial_vector adj_34 = 0;
    spatial_matrix adj_35 = 0;
    spatial_vector adj_36 = 0;
    spatial_vector adj_37 = 0;
    spatial_vector adj_38 = 0;
    spatial_vector adj_39 = 0;
    spatial_vector adj_40 = 0;
    //---------
    // forward
    var_0 = df::load(var_joint_type, var_i);
    var_1 = df::load(var_joint_axis, var_i);
    var_2 = df::load(var_joint_parent, var_i);
    var_3 = df::load(var_joint_qd_start, var_i);
    var_4 = df::load(var_body_X_sc, var_i);
    var_5 = df::spatial_transform_identity();
    var_7 = (var_2 >= var_6);
    if (var_7) {
    	var_8 = df::load(var_body_X_sc, var_2);
    }
    var_9 = df::select(var_7, var_5, var_8);
    var_10 = df::load(var_joint_X_pj, var_i);
    var_11 = df::spatial_transform_multiply(var_9, var_10);
    var_12 = jcalc_motion_cpu_func(var_0, var_1, var_11, var_joint_S_s, var_joint_qd, var_3);
    var_13 = df::spatial_vector();
    var_14 = df::spatial_vector();
    var_15 = (var_2 >= var_6);
    if (var_15) {
    	var_16 = df::load(var_body_v_s, var_2);
    	var_17 = df::load(var_body_a_s, var_2);
    }
    var_18 = df::select(var_15, var_13, var_16);
    var_19 = df::select(var_15, var_14, var_17);
    var_20 = df::add(var_18, var_12);
    var_21 = df::spatial_cross(var_20, var_12);
    var_22 = df::add(var_19, var_21);
    var_23 = df::load(var_body_X_sm, var_i);
    var_24 = df::load(var_body_I_m, var_i);
    var_25 = df::load(var_gravity, var_6);
    var_27 = df::index(var_24, var_26, var_26);
    var_28 = df::float3();
    var_29 = df::spatial_vector(var_28, var_25);
    var_30 = df::mul(var_29, var_27);
    var_31 = df::spatial_transform_get_translation(var_23);
    var_32 = df::quat_identity();
    var_33 = df::spatial_transform(var_31, var_32);
    var_34 = spatial_transform_wrench_cpu_func(var_33, var_30);
    var_35 = spatial_transform_inertia_cpu_func(var_23, var_24);
    var_36 = df::mul(var_35, var_22);
    var_37 = df::mul(var_35, var_20);
    var_38 = df::spatial_cross_dual(var_20, var_37);
    var_39 = df::add(var_36, var_38);
    df::store(var_body_v_s, var_i, var_20);
    df::store(var_body_a_s, var_i, var_22);
    var_40 = df::sub(var_39, var_34);
    df::store(var_body_f_s, var_i, var_40);
    df::store(var_body_I_s, var_i, var_35);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_6 += adj_ret;
    df::adj_store(var_body_I_s, var_i, var_35, adj_body_I_s, adj_i, adj_35);
    df::adj_store(var_body_f_s, var_i, var_40, adj_body_f_s, adj_i, adj_40);
    df::adj_sub(var_39, var_34, adj_39, adj_34, adj_40);
    df::adj_store(var_body_a_s, var_i, var_22, adj_body_a_s, adj_i, adj_22);
    df::adj_store(var_body_v_s, var_i, var_20, adj_body_v_s, adj_i, adj_20);
    df::adj_add(var_36, var_38, adj_36, adj_38, adj_39);
    df::adj_spatial_cross_dual(var_20, var_37, adj_20, adj_37, adj_38);
    df::adj_mul(var_35, var_20, adj_35, adj_20, adj_37);
    df::adj_mul(var_35, var_22, adj_35, adj_22, adj_36);
    adj_spatial_transform_inertia_cpu_func(var_23, var_24, adj_23, adj_24, adj_35);
    adj_spatial_transform_wrench_cpu_func(var_33, var_30, adj_33, adj_30, adj_34);
    df::adj_spatial_transform(var_31, var_32, adj_31, adj_32, adj_33);
    df::adj_spatial_transform_get_translation(var_23, adj_23, adj_31);
    df::adj_mul(var_29, var_27, adj_29, adj_27, adj_30);
    df::adj_spatial_vector(var_28, var_25, adj_28, adj_25, adj_29);
    df::adj_index(var_24, var_26, var_26, adj_24, adj_26, adj_26, adj_27);
    df::adj_load(var_gravity, var_6, adj_gravity, adj_6, adj_25);
    df::adj_load(var_body_I_m, var_i, adj_body_I_m, adj_i, adj_24);
    df::adj_load(var_body_X_sm, var_i, adj_body_X_sm, adj_i, adj_23);
    df::adj_add(var_19, var_21, adj_19, adj_21, adj_22);
    df::adj_spatial_cross(var_20, var_12, adj_20, adj_12, adj_21);
    df::adj_add(var_18, var_12, adj_18, adj_12, adj_20);
    df::adj_select(var_15, var_14, var_17, adj_15, adj_14, adj_17, adj_19);
    df::adj_select(var_15, var_13, var_16, adj_15, adj_13, adj_16, adj_18);
    if (var_15) {
    	df::adj_load(var_body_a_s, var_2, adj_body_a_s, adj_2, adj_17);
    	df::adj_load(var_body_v_s, var_2, adj_body_v_s, adj_2, adj_16);
    }
    adj_jcalc_motion_cpu_func(var_0, var_1, var_11, var_joint_S_s, var_joint_qd, var_3, adj_0, adj_1, adj_11, adj_joint_S_s, adj_joint_qd, adj_3, adj_12);
    df::adj_spatial_transform_multiply(var_9, var_10, adj_9, adj_10, adj_11);
    df::adj_load(var_joint_X_pj, var_i, adj_joint_X_pj, adj_i, adj_10);
    df::adj_select(var_7, var_5, var_8, adj_7, adj_5, adj_8, adj_9);
    if (var_7) {
    	df::adj_load(var_body_X_sc, var_2, adj_body_X_sc, adj_2, adj_8);
    }
    df::adj_load(var_body_X_sc, var_i, adj_body_X_sc, adj_i, adj_4);
    df::adj_load(var_joint_qd_start, var_i, adj_joint_qd_start, adj_i, adj_3);
    df::adj_load(var_joint_parent, var_i, adj_joint_parent, adj_i, adj_2);
    df::adj_load(var_joint_axis, var_i, adj_joint_axis, adj_i, adj_1);
    df::adj_load(var_joint_type, var_i, adj_joint_type, adj_i, adj_0);
    return;

}


int compute_link_tau_cpu_func(
	int var_offset,
	int var_joint_end,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_act,
	float* var_joint_target,
	float* var_joint_target_ke,
	float* var_joint_target_kd,
	float* var_joint_limit_lower,
	float* var_joint_limit_upper,
	float* var_joint_limit_ke,
	float* var_joint_limit_kd,
	spatial_vector* var_joint_S_s,
	spatial_vector* var_body_fb_s,
	spatial_vector* var_body_ft_s,
	float* var_tau)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 1;
    int var_2;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    float var_7;
    float var_8;
    float var_9;
    float var_10;
    spatial_vector var_11;
    spatial_vector var_12;
    spatial_vector var_13;
    int var_14;
    const int var_15 = 0;
    bool var_16;
    //---------
    // forward
    var_0 = df::sub(var_joint_end, var_offset);
    var_2 = df::sub(var_0, var_1);
    var_3 = df::load(var_joint_type, var_2);
    var_4 = df::load(var_joint_parent, var_2);
    var_5 = df::load(var_joint_qd_start, var_2);
    var_6 = df::load(var_joint_q_start, var_2);
    var_7 = df::load(var_joint_target_ke, var_2);
    var_8 = df::load(var_joint_target_kd, var_2);
    var_9 = df::load(var_joint_limit_ke, var_2);
    var_10 = df::load(var_joint_limit_kd, var_2);
    var_11 = df::load(var_body_fb_s, var_2);
    var_12 = df::load(var_body_ft_s, var_2);
    var_13 = df::add(var_11, var_12);
    var_14 = jcalc_tau_cpu_func(var_3, var_7, var_8, var_9, var_10, var_joint_S_s, var_joint_q, var_joint_qd, var_joint_act, var_joint_target, var_joint_limit_lower, var_joint_limit_upper, var_6, var_5, var_13, var_tau);
    var_16 = (var_4 >= var_15);
    if (var_16) {
    	df::atomic_add(var_body_ft_s, var_4, var_13);
    }
    return var_15;
    

}

void adj_compute_link_tau_cpu_func(
	int var_offset,
	int var_joint_end,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_act,
	float* var_joint_target,
	float* var_joint_target_ke,
	float* var_joint_target_kd,
	float* var_joint_limit_lower,
	float* var_joint_limit_upper,
	float* var_joint_limit_ke,
	float* var_joint_limit_kd,
	spatial_vector* var_joint_S_s,
	spatial_vector* var_body_fb_s,
	spatial_vector* var_body_ft_s,
	float* var_tau, 
	int & adj_offset,
	int & adj_joint_end,
	int* adj_joint_type,
	int* adj_joint_parent,
	int* adj_joint_q_start,
	int* adj_joint_qd_start,
	float* adj_joint_q,
	float* adj_joint_qd,
	float* adj_joint_act,
	float* adj_joint_target,
	float* adj_joint_target_ke,
	float* adj_joint_target_kd,
	float* adj_joint_limit_lower,
	float* adj_joint_limit_upper,
	float* adj_joint_limit_ke,
	float* adj_joint_limit_kd,
	spatial_vector* adj_joint_S_s,
	spatial_vector* adj_body_fb_s,
	spatial_vector* adj_body_ft_s,
	float* adj_tau,
	int & adj_ret)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 1;
    int var_2;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    float var_7;
    float var_8;
    float var_9;
    float var_10;
    spatial_vector var_11;
    spatial_vector var_12;
    spatial_vector var_13;
    int var_14;
    const int var_15 = 0;
    bool var_16;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    float adj_10 = 0;
    spatial_vector adj_11 = 0;
    spatial_vector adj_12 = 0;
    spatial_vector adj_13 = 0;
    int adj_14 = 0;
    int adj_15 = 0;
    bool adj_16 = 0;
    //---------
    // forward
    var_0 = df::sub(var_joint_end, var_offset);
    var_2 = df::sub(var_0, var_1);
    var_3 = df::load(var_joint_type, var_2);
    var_4 = df::load(var_joint_parent, var_2);
    var_5 = df::load(var_joint_qd_start, var_2);
    var_6 = df::load(var_joint_q_start, var_2);
    var_7 = df::load(var_joint_target_ke, var_2);
    var_8 = df::load(var_joint_target_kd, var_2);
    var_9 = df::load(var_joint_limit_ke, var_2);
    var_10 = df::load(var_joint_limit_kd, var_2);
    var_11 = df::load(var_body_fb_s, var_2);
    var_12 = df::load(var_body_ft_s, var_2);
    var_13 = df::add(var_11, var_12);
    var_14 = jcalc_tau_cpu_func(var_3, var_7, var_8, var_9, var_10, var_joint_S_s, var_joint_q, var_joint_qd, var_joint_act, var_joint_target, var_joint_limit_lower, var_joint_limit_upper, var_6, var_5, var_13, var_tau);
    var_16 = (var_4 >= var_15);
    if (var_16) {
    	df::atomic_add(var_body_ft_s, var_4, var_13);
    }
    goto label0;
    //---------
    // reverse
    label0:;
    adj_15 += adj_ret;
    if (var_16) {
    	df::adj_atomic_add(var_body_ft_s, var_4, var_13, adj_body_ft_s, adj_4, adj_13);
    }
    adj_jcalc_tau_cpu_func(var_3, var_7, var_8, var_9, var_10, var_joint_S_s, var_joint_q, var_joint_qd, var_joint_act, var_joint_target, var_joint_limit_lower, var_joint_limit_upper, var_6, var_5, var_13, var_tau, adj_3, adj_7, adj_8, adj_9, adj_10, adj_joint_S_s, adj_joint_q, adj_joint_qd, adj_joint_act, adj_joint_target, adj_joint_limit_lower, adj_joint_limit_upper, adj_6, adj_5, adj_13, adj_tau, adj_14);
    df::adj_add(var_11, var_12, adj_11, adj_12, adj_13);
    df::adj_load(var_body_ft_s, var_2, adj_body_ft_s, adj_2, adj_12);
    df::adj_load(var_body_fb_s, var_2, adj_body_fb_s, adj_2, adj_11);
    df::adj_load(var_joint_limit_kd, var_2, adj_joint_limit_kd, adj_2, adj_10);
    df::adj_load(var_joint_limit_ke, var_2, adj_joint_limit_ke, adj_2, adj_9);
    df::adj_load(var_joint_target_kd, var_2, adj_joint_target_kd, adj_2, adj_8);
    df::adj_load(var_joint_target_ke, var_2, adj_joint_target_ke, adj_2, adj_7);
    df::adj_load(var_joint_q_start, var_2, adj_joint_q_start, adj_2, adj_6);
    df::adj_load(var_joint_qd_start, var_2, adj_joint_qd_start, adj_2, adj_5);
    df::adj_load(var_joint_parent, var_2, adj_joint_parent, adj_2, adj_4);
    df::adj_load(var_joint_type, var_2, adj_joint_type, adj_2, adj_3);
    df::adj_sub(var_0, var_1, adj_0, adj_1, adj_2);
    df::adj_sub(var_joint_end, var_offset, adj_joint_end, adj_offset, adj_0);
    return;

}



void integrate_particles_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	df::float3* var_f,
	float* var_w,
	df::float3* var_gravity,
	float var_dt,
	df::float3* var_x_new,
	df::float3* var_v_new)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    float var_4;
    const int var_5 = 0;
    df::float3 var_6;
    df::float3 var_7;
    const float var_8 = 0.0;
    float var_9;
    float var_10;
    df::float3 var_11;
    df::float3 var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x, var_0);
    var_2 = df::load(var_v, var_0);
    var_3 = df::load(var_f, var_0);
    var_4 = df::load(var_w, var_0);
    var_6 = df::load(var_gravity, var_5);
    var_7 = df::mul(var_3, var_4);
    var_9 = df::sub(var_8, var_4);
    var_10 = df::step(var_9);
    var_11 = df::mul(var_6, var_10);
    var_12 = df::add(var_7, var_11);
    var_13 = df::mul(var_12, var_dt);
    var_14 = df::add(var_2, var_13);
    var_15 = df::mul(var_14, var_dt);
    var_16 = df::add(var_1, var_15);
    df::store(var_x_new, var_0, var_16);
    df::store(var_v_new, var_0, var_14);
    

}

void integrate_particles_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	df::float3* var_f,
	float* var_w,
	df::float3* var_gravity,
	float var_dt,
	df::float3* var_x_new,
	df::float3* var_v_new, 
	df::float3* adj_x,
	df::float3* adj_v,
	df::float3* adj_f,
	float* adj_w,
	df::float3* adj_gravity,
	float adj_dt,
	df::float3* adj_x_new,
	df::float3* adj_v_new)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    float var_4;
    const int var_5 = 0;
    df::float3 var_6;
    df::float3 var_7;
    const float var_8 = 0.0;
    float var_9;
    float var_10;
    df::float3 var_11;
    df::float3 var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    //---------
    // dual vars
    int adj_0 = 0;
    df::float3 adj_1 = 0;
    df::float3 adj_2 = 0;
    df::float3 adj_3 = 0;
    float adj_4 = 0;
    int adj_5 = 0;
    df::float3 adj_6 = 0;
    df::float3 adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    float adj_10 = 0;
    df::float3 adj_11 = 0;
    df::float3 adj_12 = 0;
    df::float3 adj_13 = 0;
    df::float3 adj_14 = 0;
    df::float3 adj_15 = 0;
    df::float3 adj_16 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x, var_0);
    var_2 = df::load(var_v, var_0);
    var_3 = df::load(var_f, var_0);
    var_4 = df::load(var_w, var_0);
    var_6 = df::load(var_gravity, var_5);
    var_7 = df::mul(var_3, var_4);
    var_9 = df::sub(var_8, var_4);
    var_10 = df::step(var_9);
    var_11 = df::mul(var_6, var_10);
    var_12 = df::add(var_7, var_11);
    var_13 = df::mul(var_12, var_dt);
    var_14 = df::add(var_2, var_13);
    var_15 = df::mul(var_14, var_dt);
    var_16 = df::add(var_1, var_15);
    df::store(var_x_new, var_0, var_16);
    df::store(var_v_new, var_0, var_14);
    //---------
    // reverse
    df::adj_store(var_v_new, var_0, var_14, adj_v_new, adj_0, adj_14);
    df::adj_store(var_x_new, var_0, var_16, adj_x_new, adj_0, adj_16);
    df::adj_add(var_1, var_15, adj_1, adj_15, adj_16);
    df::adj_mul(var_14, var_dt, adj_14, adj_dt, adj_15);
    df::adj_add(var_2, var_13, adj_2, adj_13, adj_14);
    df::adj_mul(var_12, var_dt, adj_12, adj_dt, adj_13);
    df::adj_add(var_7, var_11, adj_7, adj_11, adj_12);
    df::adj_mul(var_6, var_10, adj_6, adj_10, adj_11);
    df::adj_step(var_9, adj_9, adj_10);
    df::adj_sub(var_8, var_4, adj_8, adj_4, adj_9);
    df::adj_mul(var_3, var_4, adj_3, adj_4, adj_7);
    df::adj_load(var_gravity, var_5, adj_gravity, adj_5, adj_6);
    df::adj_load(var_w, var_0, adj_w, adj_0, adj_4);
    df::adj_load(var_f, var_0, adj_f, adj_0, adj_3);
    df::adj_load(var_v, var_0, adj_v, adj_0, adj_2);
    df::adj_load(var_x, var_0, adj_x, adj_0, adj_1);
    return;

}



// Python entry points
void integrate_particles_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_f,
	torch::Tensor var_w,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_x_new,
	torch::Tensor var_v_new)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        integrate_particles_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<df::float3*>(var_f),
			cast<float*>(var_w),
			cast<df::float3*>(var_gravity),
			var_dt,
			cast<df::float3*>(var_x_new),
			cast<df::float3*>(var_v_new));
    }
}

void integrate_particles_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_f,
	torch::Tensor var_w,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_x_new,
	torch::Tensor var_v_new, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_f,
	torch::Tensor adj_w,
	torch::Tensor adj_gravity,
	float adj_dt,
	torch::Tensor adj_x_new,
	torch::Tensor adj_v_new)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        integrate_particles_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<df::float3*>(var_f),
			cast<float*>(var_w),
			cast<df::float3*>(var_gravity),
			var_dt,
			cast<df::float3*>(var_x_new),
			cast<df::float3*>(var_v_new), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<df::float3*>(adj_f),
			cast<float*>(adj_w),
			cast<df::float3*>(adj_gravity),
			adj_dt,
			cast<df::float3*>(adj_x_new),
			cast<df::float3*>(adj_v_new));
    }
}



// Python entry points
void integrate_particles_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_f,
	torch::Tensor var_w,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_x_new,
	torch::Tensor var_v_new);

void integrate_particles_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_f,
	torch::Tensor var_w,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_x_new,
	torch::Tensor var_v_new, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_f,
	torch::Tensor adj_w,
	torch::Tensor adj_gravity,
	float adj_dt,
	torch::Tensor adj_x_new,
	torch::Tensor adj_v_new);


void integrate_rigids_cpu_kernel_forward(
	df::float3* var_rigid_x,
	quat* var_rigid_r,
	df::float3* var_rigid_v,
	df::float3* var_rigid_w,
	df::float3* var_rigid_f,
	df::float3* var_rigid_t,
	float* var_inv_m,
	mat33* var_inv_I,
	df::float3* var_gravity,
	float var_dt,
	df::float3* var_rigid_x_new,
	quat* var_rigid_r_new,
	df::float3* var_rigid_v_new,
	df::float3* var_rigid_w_new)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    quat var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    float var_7;
    mat33 var_8;
    const int var_9 = 0;
    df::float3 var_10;
    df::float3 var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    df::float3 var_22;
    df::float3 var_23;
    df::float3 var_24;
    const float var_25 = 0.0;
    quat var_26;
    quat var_27;
    const float var_28 = 0.5;
    quat var_29;
    quat var_30;
    quat var_31;
    quat var_32;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_rigid_x, var_0);
    var_2 = df::load(var_rigid_r, var_0);
    var_3 = df::load(var_rigid_v, var_0);
    var_4 = df::load(var_rigid_w, var_0);
    var_5 = df::load(var_rigid_f, var_0);
    var_6 = df::load(var_rigid_t, var_0);
    var_7 = df::load(var_inv_m, var_0);
    var_8 = df::load(var_inv_I, var_0);
    var_10 = df::load(var_gravity, var_9);
    var_11 = df::mul(var_5, var_7);
    var_12 = df::nonzero(var_7);
    var_13 = df::mul(var_10, var_12);
    var_14 = df::add(var_11, var_13);
    var_15 = df::mul(var_14, var_dt);
    var_16 = df::add(var_3, var_15);
    var_17 = df::mul(var_16, var_dt);
    var_18 = df::add(var_1, var_17);
    var_19 = df::rotate_inv(var_2, var_4);
    var_20 = df::rotate_inv(var_2, var_6);
    var_21 = df::mul(var_8, var_20);
    var_22 = df::mul(var_21, var_dt);
    var_23 = df::add(var_19, var_22);
    var_24 = df::rotate(var_2, var_23);
    var_26 = df::quat(var_24, var_25);
    var_27 = df::mul(var_26, var_2);
    var_29 = df::mul(var_27, var_28);
    var_30 = df::mul(var_29, var_dt);
    var_31 = df::add(var_2, var_30);
    var_32 = df::normalize(var_31);
    df::store(var_rigid_x_new, var_0, var_18);
    df::store(var_rigid_r_new, var_0, var_32);
    df::store(var_rigid_v_new, var_0, var_16);
    df::store(var_rigid_w_new, var_0, var_24);
    

}

void integrate_rigids_cpu_kernel_backward(
	df::float3* var_rigid_x,
	quat* var_rigid_r,
	df::float3* var_rigid_v,
	df::float3* var_rigid_w,
	df::float3* var_rigid_f,
	df::float3* var_rigid_t,
	float* var_inv_m,
	mat33* var_inv_I,
	df::float3* var_gravity,
	float var_dt,
	df::float3* var_rigid_x_new,
	quat* var_rigid_r_new,
	df::float3* var_rigid_v_new,
	df::float3* var_rigid_w_new, 
	df::float3* adj_rigid_x,
	quat* adj_rigid_r,
	df::float3* adj_rigid_v,
	df::float3* adj_rigid_w,
	df::float3* adj_rigid_f,
	df::float3* adj_rigid_t,
	float* adj_inv_m,
	mat33* adj_inv_I,
	df::float3* adj_gravity,
	float adj_dt,
	df::float3* adj_rigid_x_new,
	quat* adj_rigid_r_new,
	df::float3* adj_rigid_v_new,
	df::float3* adj_rigid_w_new)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    quat var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    float var_7;
    mat33 var_8;
    const int var_9 = 0;
    df::float3 var_10;
    df::float3 var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    df::float3 var_22;
    df::float3 var_23;
    df::float3 var_24;
    const float var_25 = 0.0;
    quat var_26;
    quat var_27;
    const float var_28 = 0.5;
    quat var_29;
    quat var_30;
    quat var_31;
    quat var_32;
    //---------
    // dual vars
    int adj_0 = 0;
    df::float3 adj_1 = 0;
    quat adj_2 = 0;
    df::float3 adj_3 = 0;
    df::float3 adj_4 = 0;
    df::float3 adj_5 = 0;
    df::float3 adj_6 = 0;
    float adj_7 = 0;
    mat33 adj_8 = 0;
    int adj_9 = 0;
    df::float3 adj_10 = 0;
    df::float3 adj_11 = 0;
    float adj_12 = 0;
    df::float3 adj_13 = 0;
    df::float3 adj_14 = 0;
    df::float3 adj_15 = 0;
    df::float3 adj_16 = 0;
    df::float3 adj_17 = 0;
    df::float3 adj_18 = 0;
    df::float3 adj_19 = 0;
    df::float3 adj_20 = 0;
    df::float3 adj_21 = 0;
    df::float3 adj_22 = 0;
    df::float3 adj_23 = 0;
    df::float3 adj_24 = 0;
    float adj_25 = 0;
    quat adj_26 = 0;
    quat adj_27 = 0;
    float adj_28 = 0;
    quat adj_29 = 0;
    quat adj_30 = 0;
    quat adj_31 = 0;
    quat adj_32 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_rigid_x, var_0);
    var_2 = df::load(var_rigid_r, var_0);
    var_3 = df::load(var_rigid_v, var_0);
    var_4 = df::load(var_rigid_w, var_0);
    var_5 = df::load(var_rigid_f, var_0);
    var_6 = df::load(var_rigid_t, var_0);
    var_7 = df::load(var_inv_m, var_0);
    var_8 = df::load(var_inv_I, var_0);
    var_10 = df::load(var_gravity, var_9);
    var_11 = df::mul(var_5, var_7);
    var_12 = df::nonzero(var_7);
    var_13 = df::mul(var_10, var_12);
    var_14 = df::add(var_11, var_13);
    var_15 = df::mul(var_14, var_dt);
    var_16 = df::add(var_3, var_15);
    var_17 = df::mul(var_16, var_dt);
    var_18 = df::add(var_1, var_17);
    var_19 = df::rotate_inv(var_2, var_4);
    var_20 = df::rotate_inv(var_2, var_6);
    var_21 = df::mul(var_8, var_20);
    var_22 = df::mul(var_21, var_dt);
    var_23 = df::add(var_19, var_22);
    var_24 = df::rotate(var_2, var_23);
    var_26 = df::quat(var_24, var_25);
    var_27 = df::mul(var_26, var_2);
    var_29 = df::mul(var_27, var_28);
    var_30 = df::mul(var_29, var_dt);
    var_31 = df::add(var_2, var_30);
    var_32 = df::normalize(var_31);
    df::store(var_rigid_x_new, var_0, var_18);
    df::store(var_rigid_r_new, var_0, var_32);
    df::store(var_rigid_v_new, var_0, var_16);
    df::store(var_rigid_w_new, var_0, var_24);
    //---------
    // reverse
    df::adj_store(var_rigid_w_new, var_0, var_24, adj_rigid_w_new, adj_0, adj_24);
    df::adj_store(var_rigid_v_new, var_0, var_16, adj_rigid_v_new, adj_0, adj_16);
    df::adj_store(var_rigid_r_new, var_0, var_32, adj_rigid_r_new, adj_0, adj_32);
    df::adj_store(var_rigid_x_new, var_0, var_18, adj_rigid_x_new, adj_0, adj_18);
    df::adj_normalize(var_31, adj_31, adj_32);
    df::adj_add(var_2, var_30, adj_2, adj_30, adj_31);
    df::adj_mul(var_29, var_dt, adj_29, adj_dt, adj_30);
    df::adj_mul(var_27, var_28, adj_27, adj_28, adj_29);
    df::adj_mul(var_26, var_2, adj_26, adj_2, adj_27);
    df::adj_quat(var_24, var_25, adj_24, adj_25, adj_26);
    df::adj_rotate(var_2, var_23, adj_2, adj_23, adj_24);
    df::adj_add(var_19, var_22, adj_19, adj_22, adj_23);
    df::adj_mul(var_21, var_dt, adj_21, adj_dt, adj_22);
    df::adj_mul(var_8, var_20, adj_8, adj_20, adj_21);
    df::adj_rotate_inv(var_2, var_6, adj_2, adj_6, adj_20);
    df::adj_rotate_inv(var_2, var_4, adj_2, adj_4, adj_19);
    df::adj_add(var_1, var_17, adj_1, adj_17, adj_18);
    df::adj_mul(var_16, var_dt, adj_16, adj_dt, adj_17);
    df::adj_add(var_3, var_15, adj_3, adj_15, adj_16);
    df::adj_mul(var_14, var_dt, adj_14, adj_dt, adj_15);
    df::adj_add(var_11, var_13, adj_11, adj_13, adj_14);
    df::adj_mul(var_10, var_12, adj_10, adj_12, adj_13);
    df::adj_nonzero(var_7, adj_7, adj_12);
    df::adj_mul(var_5, var_7, adj_5, adj_7, adj_11);
    df::adj_load(var_gravity, var_9, adj_gravity, adj_9, adj_10);
    df::adj_load(var_inv_I, var_0, adj_inv_I, adj_0, adj_8);
    df::adj_load(var_inv_m, var_0, adj_inv_m, adj_0, adj_7);
    df::adj_load(var_rigid_t, var_0, adj_rigid_t, adj_0, adj_6);
    df::adj_load(var_rigid_f, var_0, adj_rigid_f, adj_0, adj_5);
    df::adj_load(var_rigid_w, var_0, adj_rigid_w, adj_0, adj_4);
    df::adj_load(var_rigid_v, var_0, adj_rigid_v, adj_0, adj_3);
    df::adj_load(var_rigid_r, var_0, adj_rigid_r, adj_0, adj_2);
    df::adj_load(var_rigid_x, var_0, adj_rigid_x, adj_0, adj_1);
    return;

}



// Python entry points
void integrate_rigids_cpu_forward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t,
	torch::Tensor var_inv_m,
	torch::Tensor var_inv_I,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_rigid_x_new,
	torch::Tensor var_rigid_r_new,
	torch::Tensor var_rigid_v_new,
	torch::Tensor var_rigid_w_new)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        integrate_rigids_cpu_kernel_forward(
			cast<df::float3*>(var_rigid_x),
			cast<quat*>(var_rigid_r),
			cast<df::float3*>(var_rigid_v),
			cast<df::float3*>(var_rigid_w),
			cast<df::float3*>(var_rigid_f),
			cast<df::float3*>(var_rigid_t),
			cast<float*>(var_inv_m),
			cast<mat33*>(var_inv_I),
			cast<df::float3*>(var_gravity),
			var_dt,
			cast<df::float3*>(var_rigid_x_new),
			cast<quat*>(var_rigid_r_new),
			cast<df::float3*>(var_rigid_v_new),
			cast<df::float3*>(var_rigid_w_new));
    }
}

void integrate_rigids_cpu_backward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t,
	torch::Tensor var_inv_m,
	torch::Tensor var_inv_I,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_rigid_x_new,
	torch::Tensor var_rigid_r_new,
	torch::Tensor var_rigid_v_new,
	torch::Tensor var_rigid_w_new, 
	torch::Tensor adj_rigid_x,
	torch::Tensor adj_rigid_r,
	torch::Tensor adj_rigid_v,
	torch::Tensor adj_rigid_w,
	torch::Tensor adj_rigid_f,
	torch::Tensor adj_rigid_t,
	torch::Tensor adj_inv_m,
	torch::Tensor adj_inv_I,
	torch::Tensor adj_gravity,
	float adj_dt,
	torch::Tensor adj_rigid_x_new,
	torch::Tensor adj_rigid_r_new,
	torch::Tensor adj_rigid_v_new,
	torch::Tensor adj_rigid_w_new)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        integrate_rigids_cpu_kernel_backward(
			cast<df::float3*>(var_rigid_x),
			cast<quat*>(var_rigid_r),
			cast<df::float3*>(var_rigid_v),
			cast<df::float3*>(var_rigid_w),
			cast<df::float3*>(var_rigid_f),
			cast<df::float3*>(var_rigid_t),
			cast<float*>(var_inv_m),
			cast<mat33*>(var_inv_I),
			cast<df::float3*>(var_gravity),
			var_dt,
			cast<df::float3*>(var_rigid_x_new),
			cast<quat*>(var_rigid_r_new),
			cast<df::float3*>(var_rigid_v_new),
			cast<df::float3*>(var_rigid_w_new), 
			cast<df::float3*>(adj_rigid_x),
			cast<quat*>(adj_rigid_r),
			cast<df::float3*>(adj_rigid_v),
			cast<df::float3*>(adj_rigid_w),
			cast<df::float3*>(adj_rigid_f),
			cast<df::float3*>(adj_rigid_t),
			cast<float*>(adj_inv_m),
			cast<mat33*>(adj_inv_I),
			cast<df::float3*>(adj_gravity),
			adj_dt,
			cast<df::float3*>(adj_rigid_x_new),
			cast<quat*>(adj_rigid_r_new),
			cast<df::float3*>(adj_rigid_v_new),
			cast<df::float3*>(adj_rigid_w_new));
    }
}



// Python entry points
void integrate_rigids_cpu_forward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t,
	torch::Tensor var_inv_m,
	torch::Tensor var_inv_I,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_rigid_x_new,
	torch::Tensor var_rigid_r_new,
	torch::Tensor var_rigid_v_new,
	torch::Tensor var_rigid_w_new);

void integrate_rigids_cpu_backward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t,
	torch::Tensor var_inv_m,
	torch::Tensor var_inv_I,
	torch::Tensor var_gravity,
	float var_dt,
	torch::Tensor var_rigid_x_new,
	torch::Tensor var_rigid_r_new,
	torch::Tensor var_rigid_v_new,
	torch::Tensor var_rigid_w_new, 
	torch::Tensor adj_rigid_x,
	torch::Tensor adj_rigid_r,
	torch::Tensor adj_rigid_v,
	torch::Tensor adj_rigid_w,
	torch::Tensor adj_rigid_f,
	torch::Tensor adj_rigid_t,
	torch::Tensor adj_inv_m,
	torch::Tensor adj_inv_I,
	torch::Tensor adj_gravity,
	float adj_dt,
	torch::Tensor adj_rigid_x_new,
	torch::Tensor adj_rigid_r_new,
	torch::Tensor adj_rigid_v_new,
	torch::Tensor adj_rigid_w_new);


void eval_springs_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_spring_indices,
	float* var_spring_rest_lengths,
	float* var_spring_stiffness,
	float* var_spring_damping,
	df::float3* var_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 2;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    float var_10;
    float var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    float var_19;
    const float var_20 = 1.0;
    float var_21;
    df::float3 var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    df::float3 var_28;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_spring_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_spring_indices, var_8);
    var_10 = df::load(var_spring_stiffness, var_0);
    var_11 = df::load(var_spring_damping, var_0);
    var_12 = df::load(var_spring_rest_lengths, var_0);
    var_13 = df::load(var_x, var_5);
    var_14 = df::load(var_x, var_9);
    var_15 = df::load(var_v, var_5);
    var_16 = df::load(var_v, var_9);
    var_17 = df::sub(var_13, var_14);
    var_18 = df::sub(var_15, var_16);
    var_19 = df::length(var_17);
    var_21 = df::div(var_20, var_19);
    var_22 = df::mul(var_17, var_21);
    var_23 = df::sub(var_19, var_12);
    var_24 = df::dot(var_22, var_18);
    var_25 = df::mul(var_10, var_23);
    var_26 = df::mul(var_11, var_24);
    var_27 = df::add(var_25, var_26);
    var_28 = df::mul(var_22, var_27);
    df::atomic_sub(var_f, var_5, var_28);
    df::atomic_add(var_f, var_9, var_28);
    

}

void eval_springs_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_spring_indices,
	float* var_spring_rest_lengths,
	float* var_spring_stiffness,
	float* var_spring_damping,
	df::float3* var_f, 
	df::float3* adj_x,
	df::float3* adj_v,
	int* adj_spring_indices,
	float* adj_spring_rest_lengths,
	float* adj_spring_stiffness,
	float* adj_spring_damping,
	df::float3* adj_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 2;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    float var_10;
    float var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    float var_19;
    const float var_20 = 1.0;
    float var_21;
    df::float3 var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    df::float3 var_28;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    float adj_10 = 0;
    float adj_11 = 0;
    float adj_12 = 0;
    df::float3 adj_13 = 0;
    df::float3 adj_14 = 0;
    df::float3 adj_15 = 0;
    df::float3 adj_16 = 0;
    df::float3 adj_17 = 0;
    df::float3 adj_18 = 0;
    float adj_19 = 0;
    float adj_20 = 0;
    float adj_21 = 0;
    df::float3 adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    float adj_26 = 0;
    float adj_27 = 0;
    df::float3 adj_28 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_spring_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_spring_indices, var_8);
    var_10 = df::load(var_spring_stiffness, var_0);
    var_11 = df::load(var_spring_damping, var_0);
    var_12 = df::load(var_spring_rest_lengths, var_0);
    var_13 = df::load(var_x, var_5);
    var_14 = df::load(var_x, var_9);
    var_15 = df::load(var_v, var_5);
    var_16 = df::load(var_v, var_9);
    var_17 = df::sub(var_13, var_14);
    var_18 = df::sub(var_15, var_16);
    var_19 = df::length(var_17);
    var_21 = df::div(var_20, var_19);
    var_22 = df::mul(var_17, var_21);
    var_23 = df::sub(var_19, var_12);
    var_24 = df::dot(var_22, var_18);
    var_25 = df::mul(var_10, var_23);
    var_26 = df::mul(var_11, var_24);
    var_27 = df::add(var_25, var_26);
    var_28 = df::mul(var_22, var_27);
    df::atomic_sub(var_f, var_5, var_28);
    df::atomic_add(var_f, var_9, var_28);
    //---------
    // reverse
    df::adj_atomic_add(var_f, var_9, var_28, adj_f, adj_9, adj_28);
    df::adj_atomic_sub(var_f, var_5, var_28, adj_f, adj_5, adj_28);
    df::adj_mul(var_22, var_27, adj_22, adj_27, adj_28);
    df::adj_add(var_25, var_26, adj_25, adj_26, adj_27);
    df::adj_mul(var_11, var_24, adj_11, adj_24, adj_26);
    df::adj_mul(var_10, var_23, adj_10, adj_23, adj_25);
    df::adj_dot(var_22, var_18, adj_22, adj_18, adj_24);
    df::adj_sub(var_19, var_12, adj_19, adj_12, adj_23);
    df::adj_mul(var_17, var_21, adj_17, adj_21, adj_22);
    df::adj_div(var_20, var_19, adj_20, adj_19, adj_21);
    df::adj_length(var_17, adj_17, adj_19);
    df::adj_sub(var_15, var_16, adj_15, adj_16, adj_18);
    df::adj_sub(var_13, var_14, adj_13, adj_14, adj_17);
    df::adj_load(var_v, var_9, adj_v, adj_9, adj_16);
    df::adj_load(var_v, var_5, adj_v, adj_5, adj_15);
    df::adj_load(var_x, var_9, adj_x, adj_9, adj_14);
    df::adj_load(var_x, var_5, adj_x, adj_5, adj_13);
    df::adj_load(var_spring_rest_lengths, var_0, adj_spring_rest_lengths, adj_0, adj_12);
    df::adj_load(var_spring_damping, var_0, adj_spring_damping, adj_0, adj_11);
    df::adj_load(var_spring_stiffness, var_0, adj_spring_stiffness, adj_0, adj_10);
    df::adj_load(var_spring_indices, var_8, adj_spring_indices, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_6);
    df::adj_load(var_spring_indices, var_4, adj_spring_indices, adj_4, adj_5);
    df::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    return;

}



// Python entry points
void eval_springs_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	torch::Tensor var_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_springs_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_spring_indices),
			cast<float*>(var_spring_rest_lengths),
			cast<float*>(var_spring_stiffness),
			cast<float*>(var_spring_damping),
			cast<df::float3*>(var_f));
    }
}

void eval_springs_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_spring_indices,
	torch::Tensor adj_spring_rest_lengths,
	torch::Tensor adj_spring_stiffness,
	torch::Tensor adj_spring_damping,
	torch::Tensor adj_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_springs_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_spring_indices),
			cast<float*>(var_spring_rest_lengths),
			cast<float*>(var_spring_stiffness),
			cast<float*>(var_spring_damping),
			cast<df::float3*>(var_f), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<int*>(adj_spring_indices),
			cast<float*>(adj_spring_rest_lengths),
			cast<float*>(adj_spring_stiffness),
			cast<float*>(adj_spring_damping),
			cast<df::float3*>(adj_f));
    }
}



// Python entry points
void eval_springs_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	torch::Tensor var_f);

void eval_springs_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_spring_indices,
	torch::Tensor adj_spring_rest_lengths,
	torch::Tensor adj_spring_stiffness,
	torch::Tensor adj_spring_damping,
	torch::Tensor adj_f);


void eval_triangles_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	mat22* var_pose,
	float* var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	df::float3* var_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 3;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    mat22 var_22;
    float var_23;
    const float var_24 = 2.0;
    float var_25;
    const float var_26 = 1.0;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    df::float3 var_32;
    float var_33;
    df::float3 var_34;
    df::float3 var_35;
    float var_36;
    df::float3 var_37;
    float var_38;
    df::float3 var_39;
    df::float3 var_40;
    float var_41;
    df::float3 var_42;
    float var_43;
    df::float3 var_44;
    df::float3 var_45;
    df::float3 var_46;
    float var_47;
    df::float3 var_48;
    float var_49;
    df::float3 var_50;
    df::float3 var_51;
    df::float3 var_52;
    float var_53;
    float var_54;
    df::float3 var_55;
    float var_56;
    const float var_57 = 0.5;
    float var_58;
    float var_59;
    float var_60;
    float var_61;
    float var_62;
    df::float3 var_63;
    df::float3 var_64;
    df::float3 var_65;
    df::float3 var_66;
    df::float3 var_67;
    df::float3 var_68;
    df::float3 var_69;
    float var_70;
    float var_71;
    float var_72;
    float var_73;
    df::float3 var_74;
    float var_75;
    float var_76;
    float var_77;
    float var_78;
    df::float3 var_79;
    df::float3 var_80;
    float var_81;
    df::float3 var_82;
    df::float3 var_83;
    df::float3 var_84;
    df::float3 var_85;
    df::float3 var_86;
    const float var_87 = 0.3333;
    df::float3 var_88;
    df::float3 var_89;
    float var_90;
    float var_91;
    float var_92;
    float var_93;
    df::float3 var_94;
    float var_95;
    const float var_96 = 1.57079;
    float var_97;
    float var_98;
    float var_99;
    float var_100;
    df::float3 var_101;
    float var_102;
    df::float3 var_103;
    df::float3 var_104;
    df::float3 var_105;
    df::float3 var_106;
    df::float3 var_107;
    df::float3 var_108;
    df::float3 var_109;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::load(var_x, var_5);
    var_15 = df::load(var_x, var_9);
    var_16 = df::load(var_x, var_13);
    var_17 = df::load(var_v, var_5);
    var_18 = df::load(var_v, var_9);
    var_19 = df::load(var_v, var_13);
    var_20 = df::sub(var_15, var_14);
    var_21 = df::sub(var_16, var_14);
    var_22 = df::load(var_pose, var_0);
    var_23 = df::determinant(var_22);
    var_25 = df::mul(var_23, var_24);
    var_27 = df::div(var_26, var_25);
    var_28 = df::mul(var_k_mu, var_27);
    var_29 = df::mul(var_k_lambda, var_27);
    var_30 = df::mul(var_k_damp, var_27);
    var_31 = df::index(var_22, var_3, var_3);
    var_32 = df::mul(var_20, var_31);
    var_33 = df::index(var_22, var_7, var_3);
    var_34 = df::mul(var_21, var_33);
    var_35 = df::add(var_32, var_34);
    var_36 = df::index(var_22, var_3, var_7);
    var_37 = df::mul(var_20, var_36);
    var_38 = df::index(var_22, var_7, var_7);
    var_39 = df::mul(var_21, var_38);
    var_40 = df::add(var_37, var_39);
    var_41 = df::index(var_22, var_3, var_3);
    var_42 = df::mul(var_35, var_41);
    var_43 = df::index(var_22, var_3, var_7);
    var_44 = df::mul(var_40, var_43);
    var_45 = df::add(var_42, var_44);
    var_46 = df::mul(var_45, var_28);
    var_47 = df::index(var_22, var_7, var_3);
    var_48 = df::mul(var_35, var_47);
    var_49 = df::index(var_22, var_7, var_7);
    var_50 = df::mul(var_40, var_49);
    var_51 = df::add(var_48, var_50);
    var_52 = df::mul(var_51, var_28);
    var_53 = df::div(var_28, var_29);
    var_54 = df::add(var_26, var_53);
    var_55 = df::cross(var_20, var_21);
    var_56 = df::length(var_55);
    var_58 = df::mul(var_56, var_57);
    var_59 = df::load(var_activation, var_0);
    var_60 = df::mul(var_58, var_25);
    var_61 = df::sub(var_60, var_54);
    var_62 = df::add(var_61, var_59);
    var_63 = df::normalize(var_55);
    var_64 = df::cross(var_21, var_63);
    var_65 = df::mul(var_64, var_25);
    var_66 = df::mul(var_65, var_57);
    var_67 = df::cross(var_63, var_20);
    var_68 = df::mul(var_67, var_25);
    var_69 = df::mul(var_68, var_57);
    var_70 = df::mul(var_29, var_62);
    var_71 = df::dot(var_66, var_18);
    var_72 = df::dot(var_69, var_19);
    var_73 = df::add(var_71, var_72);
    var_74 = df::add(var_66, var_69);
    var_75 = df::dot(var_74, var_17);
    var_76 = df::sub(var_73, var_75);
    var_77 = df::mul(var_30, var_76);
    var_78 = df::add(var_70, var_77);
    var_79 = df::mul(var_66, var_78);
    var_80 = df::add(var_46, var_79);
    var_81 = df::add(var_70, var_77);
    var_82 = df::mul(var_69, var_81);
    var_83 = df::add(var_52, var_82);
    var_84 = df::add(var_80, var_83);
    var_85 = df::add(var_17, var_19);
    var_86 = df::add(var_85, var_18);
    var_88 = df::mul(var_86, var_87);
    var_89 = df::normalize(var_88);
    var_90 = df::mul(var_k_drag, var_58);
    var_91 = df::dot(var_63, var_88);
    var_92 = df::abs(var_91);
    var_93 = df::mul(var_90, var_92);
    var_94 = df::mul(var_88, var_93);
    var_95 = df::mul(var_k_lift, var_58);
    var_97 = df::dot(var_63, var_89);
    var_98 = df::acos(var_97);
    var_99 = df::sub(var_96, var_98);
    var_100 = df::mul(var_95, var_99);
    var_101 = df::mul(var_63, var_100);
    var_102 = df::dot(var_88, var_88);
    var_103 = df::mul(var_101, var_102);
    var_104 = df::sub(var_84, var_94);
    var_105 = df::sub(var_104, var_103);
    var_106 = df::add(var_80, var_94);
    var_107 = df::add(var_106, var_103);
    var_108 = df::add(var_83, var_94);
    var_109 = df::add(var_108, var_103);
    df::atomic_add(var_f, var_5, var_105);
    df::atomic_sub(var_f, var_9, var_107);
    df::atomic_sub(var_f, var_13, var_109);
    

}

void eval_triangles_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	mat22* var_pose,
	float* var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	df::float3* var_f, 
	df::float3* adj_x,
	df::float3* adj_v,
	int* adj_indices,
	mat22* adj_pose,
	float* adj_activation,
	float adj_k_mu,
	float adj_k_lambda,
	float adj_k_damp,
	float adj_k_drag,
	float adj_k_lift,
	df::float3* adj_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 3;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    mat22 var_22;
    float var_23;
    const float var_24 = 2.0;
    float var_25;
    const float var_26 = 1.0;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    df::float3 var_32;
    float var_33;
    df::float3 var_34;
    df::float3 var_35;
    float var_36;
    df::float3 var_37;
    float var_38;
    df::float3 var_39;
    df::float3 var_40;
    float var_41;
    df::float3 var_42;
    float var_43;
    df::float3 var_44;
    df::float3 var_45;
    df::float3 var_46;
    float var_47;
    df::float3 var_48;
    float var_49;
    df::float3 var_50;
    df::float3 var_51;
    df::float3 var_52;
    float var_53;
    float var_54;
    df::float3 var_55;
    float var_56;
    const float var_57 = 0.5;
    float var_58;
    float var_59;
    float var_60;
    float var_61;
    float var_62;
    df::float3 var_63;
    df::float3 var_64;
    df::float3 var_65;
    df::float3 var_66;
    df::float3 var_67;
    df::float3 var_68;
    df::float3 var_69;
    float var_70;
    float var_71;
    float var_72;
    float var_73;
    df::float3 var_74;
    float var_75;
    float var_76;
    float var_77;
    float var_78;
    df::float3 var_79;
    df::float3 var_80;
    float var_81;
    df::float3 var_82;
    df::float3 var_83;
    df::float3 var_84;
    df::float3 var_85;
    df::float3 var_86;
    const float var_87 = 0.3333;
    df::float3 var_88;
    df::float3 var_89;
    float var_90;
    float var_91;
    float var_92;
    float var_93;
    df::float3 var_94;
    float var_95;
    const float var_96 = 1.57079;
    float var_97;
    float var_98;
    float var_99;
    float var_100;
    df::float3 var_101;
    float var_102;
    df::float3 var_103;
    df::float3 var_104;
    df::float3 var_105;
    df::float3 var_106;
    df::float3 var_107;
    df::float3 var_108;
    df::float3 var_109;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    int adj_10 = 0;
    int adj_11 = 0;
    int adj_12 = 0;
    int adj_13 = 0;
    df::float3 adj_14 = 0;
    df::float3 adj_15 = 0;
    df::float3 adj_16 = 0;
    df::float3 adj_17 = 0;
    df::float3 adj_18 = 0;
    df::float3 adj_19 = 0;
    df::float3 adj_20 = 0;
    df::float3 adj_21 = 0;
    mat22 adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    float adj_26 = 0;
    float adj_27 = 0;
    float adj_28 = 0;
    float adj_29 = 0;
    float adj_30 = 0;
    float adj_31 = 0;
    df::float3 adj_32 = 0;
    float adj_33 = 0;
    df::float3 adj_34 = 0;
    df::float3 adj_35 = 0;
    float adj_36 = 0;
    df::float3 adj_37 = 0;
    float adj_38 = 0;
    df::float3 adj_39 = 0;
    df::float3 adj_40 = 0;
    float adj_41 = 0;
    df::float3 adj_42 = 0;
    float adj_43 = 0;
    df::float3 adj_44 = 0;
    df::float3 adj_45 = 0;
    df::float3 adj_46 = 0;
    float adj_47 = 0;
    df::float3 adj_48 = 0;
    float adj_49 = 0;
    df::float3 adj_50 = 0;
    df::float3 adj_51 = 0;
    df::float3 adj_52 = 0;
    float adj_53 = 0;
    float adj_54 = 0;
    df::float3 adj_55 = 0;
    float adj_56 = 0;
    float adj_57 = 0;
    float adj_58 = 0;
    float adj_59 = 0;
    float adj_60 = 0;
    float adj_61 = 0;
    float adj_62 = 0;
    df::float3 adj_63 = 0;
    df::float3 adj_64 = 0;
    df::float3 adj_65 = 0;
    df::float3 adj_66 = 0;
    df::float3 adj_67 = 0;
    df::float3 adj_68 = 0;
    df::float3 adj_69 = 0;
    float adj_70 = 0;
    float adj_71 = 0;
    float adj_72 = 0;
    float adj_73 = 0;
    df::float3 adj_74 = 0;
    float adj_75 = 0;
    float adj_76 = 0;
    float adj_77 = 0;
    float adj_78 = 0;
    df::float3 adj_79 = 0;
    df::float3 adj_80 = 0;
    float adj_81 = 0;
    df::float3 adj_82 = 0;
    df::float3 adj_83 = 0;
    df::float3 adj_84 = 0;
    df::float3 adj_85 = 0;
    df::float3 adj_86 = 0;
    float adj_87 = 0;
    df::float3 adj_88 = 0;
    df::float3 adj_89 = 0;
    float adj_90 = 0;
    float adj_91 = 0;
    float adj_92 = 0;
    float adj_93 = 0;
    df::float3 adj_94 = 0;
    float adj_95 = 0;
    float adj_96 = 0;
    float adj_97 = 0;
    float adj_98 = 0;
    float adj_99 = 0;
    float adj_100 = 0;
    df::float3 adj_101 = 0;
    float adj_102 = 0;
    df::float3 adj_103 = 0;
    df::float3 adj_104 = 0;
    df::float3 adj_105 = 0;
    df::float3 adj_106 = 0;
    df::float3 adj_107 = 0;
    df::float3 adj_108 = 0;
    df::float3 adj_109 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::load(var_x, var_5);
    var_15 = df::load(var_x, var_9);
    var_16 = df::load(var_x, var_13);
    var_17 = df::load(var_v, var_5);
    var_18 = df::load(var_v, var_9);
    var_19 = df::load(var_v, var_13);
    var_20 = df::sub(var_15, var_14);
    var_21 = df::sub(var_16, var_14);
    var_22 = df::load(var_pose, var_0);
    var_23 = df::determinant(var_22);
    var_25 = df::mul(var_23, var_24);
    var_27 = df::div(var_26, var_25);
    var_28 = df::mul(var_k_mu, var_27);
    var_29 = df::mul(var_k_lambda, var_27);
    var_30 = df::mul(var_k_damp, var_27);
    var_31 = df::index(var_22, var_3, var_3);
    var_32 = df::mul(var_20, var_31);
    var_33 = df::index(var_22, var_7, var_3);
    var_34 = df::mul(var_21, var_33);
    var_35 = df::add(var_32, var_34);
    var_36 = df::index(var_22, var_3, var_7);
    var_37 = df::mul(var_20, var_36);
    var_38 = df::index(var_22, var_7, var_7);
    var_39 = df::mul(var_21, var_38);
    var_40 = df::add(var_37, var_39);
    var_41 = df::index(var_22, var_3, var_3);
    var_42 = df::mul(var_35, var_41);
    var_43 = df::index(var_22, var_3, var_7);
    var_44 = df::mul(var_40, var_43);
    var_45 = df::add(var_42, var_44);
    var_46 = df::mul(var_45, var_28);
    var_47 = df::index(var_22, var_7, var_3);
    var_48 = df::mul(var_35, var_47);
    var_49 = df::index(var_22, var_7, var_7);
    var_50 = df::mul(var_40, var_49);
    var_51 = df::add(var_48, var_50);
    var_52 = df::mul(var_51, var_28);
    var_53 = df::div(var_28, var_29);
    var_54 = df::add(var_26, var_53);
    var_55 = df::cross(var_20, var_21);
    var_56 = df::length(var_55);
    var_58 = df::mul(var_56, var_57);
    var_59 = df::load(var_activation, var_0);
    var_60 = df::mul(var_58, var_25);
    var_61 = df::sub(var_60, var_54);
    var_62 = df::add(var_61, var_59);
    var_63 = df::normalize(var_55);
    var_64 = df::cross(var_21, var_63);
    var_65 = df::mul(var_64, var_25);
    var_66 = df::mul(var_65, var_57);
    var_67 = df::cross(var_63, var_20);
    var_68 = df::mul(var_67, var_25);
    var_69 = df::mul(var_68, var_57);
    var_70 = df::mul(var_29, var_62);
    var_71 = df::dot(var_66, var_18);
    var_72 = df::dot(var_69, var_19);
    var_73 = df::add(var_71, var_72);
    var_74 = df::add(var_66, var_69);
    var_75 = df::dot(var_74, var_17);
    var_76 = df::sub(var_73, var_75);
    var_77 = df::mul(var_30, var_76);
    var_78 = df::add(var_70, var_77);
    var_79 = df::mul(var_66, var_78);
    var_80 = df::add(var_46, var_79);
    var_81 = df::add(var_70, var_77);
    var_82 = df::mul(var_69, var_81);
    var_83 = df::add(var_52, var_82);
    var_84 = df::add(var_80, var_83);
    var_85 = df::add(var_17, var_19);
    var_86 = df::add(var_85, var_18);
    var_88 = df::mul(var_86, var_87);
    var_89 = df::normalize(var_88);
    var_90 = df::mul(var_k_drag, var_58);
    var_91 = df::dot(var_63, var_88);
    var_92 = df::abs(var_91);
    var_93 = df::mul(var_90, var_92);
    var_94 = df::mul(var_88, var_93);
    var_95 = df::mul(var_k_lift, var_58);
    var_97 = df::dot(var_63, var_89);
    var_98 = df::acos(var_97);
    var_99 = df::sub(var_96, var_98);
    var_100 = df::mul(var_95, var_99);
    var_101 = df::mul(var_63, var_100);
    var_102 = df::dot(var_88, var_88);
    var_103 = df::mul(var_101, var_102);
    var_104 = df::sub(var_84, var_94);
    var_105 = df::sub(var_104, var_103);
    var_106 = df::add(var_80, var_94);
    var_107 = df::add(var_106, var_103);
    var_108 = df::add(var_83, var_94);
    var_109 = df::add(var_108, var_103);
    df::atomic_add(var_f, var_5, var_105);
    df::atomic_sub(var_f, var_9, var_107);
    df::atomic_sub(var_f, var_13, var_109);
    //---------
    // reverse
    df::adj_atomic_sub(var_f, var_13, var_109, adj_f, adj_13, adj_109);
    df::adj_atomic_sub(var_f, var_9, var_107, adj_f, adj_9, adj_107);
    df::adj_atomic_add(var_f, var_5, var_105, adj_f, adj_5, adj_105);
    df::adj_add(var_108, var_103, adj_108, adj_103, adj_109);
    df::adj_add(var_83, var_94, adj_83, adj_94, adj_108);
    df::adj_add(var_106, var_103, adj_106, adj_103, adj_107);
    df::adj_add(var_80, var_94, adj_80, adj_94, adj_106);
    df::adj_sub(var_104, var_103, adj_104, adj_103, adj_105);
    df::adj_sub(var_84, var_94, adj_84, adj_94, adj_104);
    df::adj_mul(var_101, var_102, adj_101, adj_102, adj_103);
    df::adj_dot(var_88, var_88, adj_88, adj_88, adj_102);
    df::adj_mul(var_63, var_100, adj_63, adj_100, adj_101);
    df::adj_mul(var_95, var_99, adj_95, adj_99, adj_100);
    df::adj_sub(var_96, var_98, adj_96, adj_98, adj_99);
    df::adj_acos(var_97, adj_97, adj_98);
    df::adj_dot(var_63, var_89, adj_63, adj_89, adj_97);
    df::adj_mul(var_k_lift, var_58, adj_k_lift, adj_58, adj_95);
    df::adj_mul(var_88, var_93, adj_88, adj_93, adj_94);
    df::adj_mul(var_90, var_92, adj_90, adj_92, adj_93);
    df::adj_abs(var_91, adj_91, adj_92);
    df::adj_dot(var_63, var_88, adj_63, adj_88, adj_91);
    df::adj_mul(var_k_drag, var_58, adj_k_drag, adj_58, adj_90);
    df::adj_normalize(var_88, adj_88, adj_89);
    df::adj_mul(var_86, var_87, adj_86, adj_87, adj_88);
    df::adj_add(var_85, var_18, adj_85, adj_18, adj_86);
    df::adj_add(var_17, var_19, adj_17, adj_19, adj_85);
    df::adj_add(var_80, var_83, adj_80, adj_83, adj_84);
    df::adj_add(var_52, var_82, adj_52, adj_82, adj_83);
    df::adj_mul(var_69, var_81, adj_69, adj_81, adj_82);
    df::adj_add(var_70, var_77, adj_70, adj_77, adj_81);
    df::adj_add(var_46, var_79, adj_46, adj_79, adj_80);
    df::adj_mul(var_66, var_78, adj_66, adj_78, adj_79);
    df::adj_add(var_70, var_77, adj_70, adj_77, adj_78);
    df::adj_mul(var_30, var_76, adj_30, adj_76, adj_77);
    df::adj_sub(var_73, var_75, adj_73, adj_75, adj_76);
    df::adj_dot(var_74, var_17, adj_74, adj_17, adj_75);
    df::adj_add(var_66, var_69, adj_66, adj_69, adj_74);
    df::adj_add(var_71, var_72, adj_71, adj_72, adj_73);
    df::adj_dot(var_69, var_19, adj_69, adj_19, adj_72);
    df::adj_dot(var_66, var_18, adj_66, adj_18, adj_71);
    df::adj_mul(var_29, var_62, adj_29, adj_62, adj_70);
    df::adj_mul(var_68, var_57, adj_68, adj_57, adj_69);
    df::adj_mul(var_67, var_25, adj_67, adj_25, adj_68);
    df::adj_cross(var_63, var_20, adj_63, adj_20, adj_67);
    df::adj_mul(var_65, var_57, adj_65, adj_57, adj_66);
    df::adj_mul(var_64, var_25, adj_64, adj_25, adj_65);
    df::adj_cross(var_21, var_63, adj_21, adj_63, adj_64);
    df::adj_normalize(var_55, adj_55, adj_63);
    df::adj_add(var_61, var_59, adj_61, adj_59, adj_62);
    df::adj_sub(var_60, var_54, adj_60, adj_54, adj_61);
    df::adj_mul(var_58, var_25, adj_58, adj_25, adj_60);
    df::adj_load(var_activation, var_0, adj_activation, adj_0, adj_59);
    df::adj_mul(var_56, var_57, adj_56, adj_57, adj_58);
    df::adj_length(var_55, adj_55, adj_56);
    df::adj_cross(var_20, var_21, adj_20, adj_21, adj_55);
    df::adj_add(var_26, var_53, adj_26, adj_53, adj_54);
    df::adj_div(var_28, var_29, adj_28, adj_29, adj_53);
    df::adj_mul(var_51, var_28, adj_51, adj_28, adj_52);
    df::adj_add(var_48, var_50, adj_48, adj_50, adj_51);
    df::adj_mul(var_40, var_49, adj_40, adj_49, adj_50);
    df::adj_index(var_22, var_7, var_7, adj_22, adj_7, adj_7, adj_49);
    df::adj_mul(var_35, var_47, adj_35, adj_47, adj_48);
    df::adj_index(var_22, var_7, var_3, adj_22, adj_7, adj_3, adj_47);
    df::adj_mul(var_45, var_28, adj_45, adj_28, adj_46);
    df::adj_add(var_42, var_44, adj_42, adj_44, adj_45);
    df::adj_mul(var_40, var_43, adj_40, adj_43, adj_44);
    df::adj_index(var_22, var_3, var_7, adj_22, adj_3, adj_7, adj_43);
    df::adj_mul(var_35, var_41, adj_35, adj_41, adj_42);
    df::adj_index(var_22, var_3, var_3, adj_22, adj_3, adj_3, adj_41);
    df::adj_add(var_37, var_39, adj_37, adj_39, adj_40);
    df::adj_mul(var_21, var_38, adj_21, adj_38, adj_39);
    df::adj_index(var_22, var_7, var_7, adj_22, adj_7, adj_7, adj_38);
    df::adj_mul(var_20, var_36, adj_20, adj_36, adj_37);
    df::adj_index(var_22, var_3, var_7, adj_22, adj_3, adj_7, adj_36);
    df::adj_add(var_32, var_34, adj_32, adj_34, adj_35);
    df::adj_mul(var_21, var_33, adj_21, adj_33, adj_34);
    df::adj_index(var_22, var_7, var_3, adj_22, adj_7, adj_3, adj_33);
    df::adj_mul(var_20, var_31, adj_20, adj_31, adj_32);
    df::adj_index(var_22, var_3, var_3, adj_22, adj_3, adj_3, adj_31);
    df::adj_mul(var_k_damp, var_27, adj_k_damp, adj_27, adj_30);
    df::adj_mul(var_k_lambda, var_27, adj_k_lambda, adj_27, adj_29);
    df::adj_mul(var_k_mu, var_27, adj_k_mu, adj_27, adj_28);
    df::adj_div(var_26, var_25, adj_26, adj_25, adj_27);
    df::adj_mul(var_23, var_24, adj_23, adj_24, adj_25);
    df::adj_determinant(var_22, adj_22, adj_23);
    df::adj_load(var_pose, var_0, adj_pose, adj_0, adj_22);
    df::adj_sub(var_16, var_14, adj_16, adj_14, adj_21);
    df::adj_sub(var_15, var_14, adj_15, adj_14, adj_20);
    df::adj_load(var_v, var_13, adj_v, adj_13, adj_19);
    df::adj_load(var_v, var_9, adj_v, adj_9, adj_18);
    df::adj_load(var_v, var_5, adj_v, adj_5, adj_17);
    df::adj_load(var_x, var_13, adj_x, adj_13, adj_16);
    df::adj_load(var_x, var_9, adj_x, adj_9, adj_15);
    df::adj_load(var_x, var_5, adj_x, adj_5, adj_14);
    df::adj_load(var_indices, var_12, adj_indices, adj_12, adj_13);
    df::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_10);
    df::adj_load(var_indices, var_8, adj_indices, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_6);
    df::adj_load(var_indices, var_4, adj_indices, adj_4, adj_5);
    df::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    return;

}



// Python entry points
void eval_triangles_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_triangles_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<mat22*>(var_pose),
			cast<float*>(var_activation),
			var_k_mu,
			var_k_lambda,
			var_k_damp,
			var_k_drag,
			var_k_lift,
			cast<df::float3*>(var_f));
    }
}

void eval_triangles_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	float adj_k_mu,
	float adj_k_lambda,
	float adj_k_damp,
	float adj_k_drag,
	float adj_k_lift,
	torch::Tensor adj_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_triangles_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<mat22*>(var_pose),
			cast<float*>(var_activation),
			var_k_mu,
			var_k_lambda,
			var_k_damp,
			var_k_drag,
			var_k_lift,
			cast<df::float3*>(var_f), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<int*>(adj_indices),
			cast<mat22*>(adj_pose),
			cast<float*>(adj_activation),
			adj_k_mu,
			adj_k_lambda,
			adj_k_damp,
			adj_k_drag,
			adj_k_lift,
			cast<df::float3*>(adj_f));
    }
}



// Python entry points
void eval_triangles_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f);

void eval_triangles_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	float adj_k_mu,
	float adj_k_lambda,
	float adj_k_damp,
	float adj_k_drag,
	float adj_k_lift,
	torch::Tensor adj_f);


void eval_triangles_contact_cpu_kernel_forward(
	int var_num_particles,
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	mat22* var_pose,
	float* var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	df::float3* var_f)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    df::float3 var_3;
    const int var_4 = 3;
    int var_5;
    const int var_6 = 0;
    int var_7;
    int var_8;
    int var_9;
    const int var_10 = 1;
    int var_11;
    int var_12;
    int var_13;
    const int var_14 = 2;
    int var_15;
    int var_16;
    bool var_17;
    bool var_18;
    bool var_19;
    bool var_20;
    df::float3 var_21;
    df::float3 var_22;
    df::float3 var_23;
    df::float3 var_24;
    float var_25;
    df::float3 var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    float var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    float var_34;
    df::float3 var_35;
    const float var_36 = 0.01;
    float var_37;
    const float var_38 = 0.0;
    float var_39;
    df::float3 var_40;
    const float var_41 = 100000.0;
    df::float3 var_42;
    float var_43;
    df::float3 var_44;
    float var_45;
    df::float3 var_46;
    float var_47;
    df::float3 var_48;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::div(var_0, var_num_particles);
    var_2 = df::mod(var_0, var_num_particles);
    var_3 = df::load(var_x, var_2);
    var_5 = df::mul(var_1, var_4);
    var_7 = df::add(var_5, var_6);
    var_8 = df::load(var_indices, var_7);
    var_9 = df::mul(var_1, var_4);
    var_11 = df::add(var_9, var_10);
    var_12 = df::load(var_indices, var_11);
    var_13 = df::mul(var_1, var_4);
    var_15 = df::add(var_13, var_14);
    var_16 = df::load(var_indices, var_15);
    var_17 = (var_8 == var_2);
    var_18 = (var_12 == var_2);
    var_19 = (var_16 == var_2);
    var_20 = var_17 || var_18 || var_19;
    if (var_20) {
    	return;
    }
    var_21 = df::load(var_x, var_8);
    var_22 = df::load(var_x, var_12);
    var_23 = df::load(var_x, var_16);
    var_24 = triangle_closest_point_barycentric_cpu_func(var_21, var_22, var_23, var_3);
    var_25 = df::index(var_24, var_6);
    var_26 = df::mul(var_21, var_25);
    var_27 = df::index(var_24, var_10);
    var_28 = df::mul(var_22, var_27);
    var_29 = df::add(var_26, var_28);
    var_30 = df::index(var_24, var_14);
    var_31 = df::mul(var_23, var_30);
    var_32 = df::add(var_29, var_31);
    var_33 = df::sub(var_3, var_32);
    var_34 = df::dot(var_33, var_33);
    var_35 = df::normalize(var_33);
    var_37 = df::sub(var_34, var_36);
    var_39 = df::min(var_37, var_38);
    var_40 = df::mul(var_35, var_39);
    var_42 = df::mul(var_40, var_41);
    df::atomic_sub(var_f, var_2, var_42);
    var_43 = df::index(var_24, var_6);
    var_44 = df::mul(var_42, var_43);
    df::atomic_add(var_f, var_8, var_44);
    var_45 = df::index(var_24, var_10);
    var_46 = df::mul(var_42, var_45);
    df::atomic_add(var_f, var_12, var_46);
    var_47 = df::index(var_24, var_14);
    var_48 = df::mul(var_42, var_47);
    df::atomic_add(var_f, var_16, var_48);
    

}

void eval_triangles_contact_cpu_kernel_backward(
	int var_num_particles,
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	mat22* var_pose,
	float* var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	df::float3* var_f, 
	int adj_num_particles,
	df::float3* adj_x,
	df::float3* adj_v,
	int* adj_indices,
	mat22* adj_pose,
	float* adj_activation,
	float adj_k_mu,
	float adj_k_lambda,
	float adj_k_damp,
	float adj_k_drag,
	float adj_k_lift,
	df::float3* adj_f)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    df::float3 var_3;
    const int var_4 = 3;
    int var_5;
    const int var_6 = 0;
    int var_7;
    int var_8;
    int var_9;
    const int var_10 = 1;
    int var_11;
    int var_12;
    int var_13;
    const int var_14 = 2;
    int var_15;
    int var_16;
    bool var_17;
    bool var_18;
    bool var_19;
    bool var_20;
    df::float3 var_21;
    df::float3 var_22;
    df::float3 var_23;
    df::float3 var_24;
    float var_25;
    df::float3 var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    float var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    float var_34;
    df::float3 var_35;
    const float var_36 = 0.01;
    float var_37;
    const float var_38 = 0.0;
    float var_39;
    df::float3 var_40;
    const float var_41 = 100000.0;
    df::float3 var_42;
    float var_43;
    df::float3 var_44;
    float var_45;
    df::float3 var_46;
    float var_47;
    df::float3 var_48;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    df::float3 adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    int adj_10 = 0;
    int adj_11 = 0;
    int adj_12 = 0;
    int adj_13 = 0;
    int adj_14 = 0;
    int adj_15 = 0;
    int adj_16 = 0;
    bool adj_17 = 0;
    bool adj_18 = 0;
    bool adj_19 = 0;
    bool adj_20 = 0;
    df::float3 adj_21 = 0;
    df::float3 adj_22 = 0;
    df::float3 adj_23 = 0;
    df::float3 adj_24 = 0;
    float adj_25 = 0;
    df::float3 adj_26 = 0;
    float adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    float adj_30 = 0;
    df::float3 adj_31 = 0;
    df::float3 adj_32 = 0;
    df::float3 adj_33 = 0;
    float adj_34 = 0;
    df::float3 adj_35 = 0;
    float adj_36 = 0;
    float adj_37 = 0;
    float adj_38 = 0;
    float adj_39 = 0;
    df::float3 adj_40 = 0;
    float adj_41 = 0;
    df::float3 adj_42 = 0;
    float adj_43 = 0;
    df::float3 adj_44 = 0;
    float adj_45 = 0;
    df::float3 adj_46 = 0;
    float adj_47 = 0;
    df::float3 adj_48 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::div(var_0, var_num_particles);
    var_2 = df::mod(var_0, var_num_particles);
    var_3 = df::load(var_x, var_2);
    var_5 = df::mul(var_1, var_4);
    var_7 = df::add(var_5, var_6);
    var_8 = df::load(var_indices, var_7);
    var_9 = df::mul(var_1, var_4);
    var_11 = df::add(var_9, var_10);
    var_12 = df::load(var_indices, var_11);
    var_13 = df::mul(var_1, var_4);
    var_15 = df::add(var_13, var_14);
    var_16 = df::load(var_indices, var_15);
    var_17 = (var_8 == var_2);
    var_18 = (var_12 == var_2);
    var_19 = (var_16 == var_2);
    var_20 = var_17 || var_18 || var_19;
    if (var_20) {
    	goto label0;
    }
    var_21 = df::load(var_x, var_8);
    var_22 = df::load(var_x, var_12);
    var_23 = df::load(var_x, var_16);
    var_24 = triangle_closest_point_barycentric_cpu_func(var_21, var_22, var_23, var_3);
    var_25 = df::index(var_24, var_6);
    var_26 = df::mul(var_21, var_25);
    var_27 = df::index(var_24, var_10);
    var_28 = df::mul(var_22, var_27);
    var_29 = df::add(var_26, var_28);
    var_30 = df::index(var_24, var_14);
    var_31 = df::mul(var_23, var_30);
    var_32 = df::add(var_29, var_31);
    var_33 = df::sub(var_3, var_32);
    var_34 = df::dot(var_33, var_33);
    var_35 = df::normalize(var_33);
    var_37 = df::sub(var_34, var_36);
    var_39 = df::min(var_37, var_38);
    var_40 = df::mul(var_35, var_39);
    var_42 = df::mul(var_40, var_41);
    df::atomic_sub(var_f, var_2, var_42);
    var_43 = df::index(var_24, var_6);
    var_44 = df::mul(var_42, var_43);
    df::atomic_add(var_f, var_8, var_44);
    var_45 = df::index(var_24, var_10);
    var_46 = df::mul(var_42, var_45);
    df::atomic_add(var_f, var_12, var_46);
    var_47 = df::index(var_24, var_14);
    var_48 = df::mul(var_42, var_47);
    df::atomic_add(var_f, var_16, var_48);
    //---------
    // reverse
    df::adj_atomic_add(var_f, var_16, var_48, adj_f, adj_16, adj_48);
    df::adj_mul(var_42, var_47, adj_42, adj_47, adj_48);
    df::adj_index(var_24, var_14, adj_24, adj_14, adj_47);
    df::adj_atomic_add(var_f, var_12, var_46, adj_f, adj_12, adj_46);
    df::adj_mul(var_42, var_45, adj_42, adj_45, adj_46);
    df::adj_index(var_24, var_10, adj_24, adj_10, adj_45);
    df::adj_atomic_add(var_f, var_8, var_44, adj_f, adj_8, adj_44);
    df::adj_mul(var_42, var_43, adj_42, adj_43, adj_44);
    df::adj_index(var_24, var_6, adj_24, adj_6, adj_43);
    df::adj_atomic_sub(var_f, var_2, var_42, adj_f, adj_2, adj_42);
    df::adj_mul(var_40, var_41, adj_40, adj_41, adj_42);
    df::adj_mul(var_35, var_39, adj_35, adj_39, adj_40);
    df::adj_min(var_37, var_38, adj_37, adj_38, adj_39);
    df::adj_sub(var_34, var_36, adj_34, adj_36, adj_37);
    df::adj_normalize(var_33, adj_33, adj_35);
    df::adj_dot(var_33, var_33, adj_33, adj_33, adj_34);
    df::adj_sub(var_3, var_32, adj_3, adj_32, adj_33);
    df::adj_add(var_29, var_31, adj_29, adj_31, adj_32);
    df::adj_mul(var_23, var_30, adj_23, adj_30, adj_31);
    df::adj_index(var_24, var_14, adj_24, adj_14, adj_30);
    df::adj_add(var_26, var_28, adj_26, adj_28, adj_29);
    df::adj_mul(var_22, var_27, adj_22, adj_27, adj_28);
    df::adj_index(var_24, var_10, adj_24, adj_10, adj_27);
    df::adj_mul(var_21, var_25, adj_21, adj_25, adj_26);
    df::adj_index(var_24, var_6, adj_24, adj_6, adj_25);
    adj_triangle_closest_point_barycentric_cpu_func(var_21, var_22, var_23, var_3, adj_21, adj_22, adj_23, adj_3, adj_24);
    df::adj_load(var_x, var_16, adj_x, adj_16, adj_23);
    df::adj_load(var_x, var_12, adj_x, adj_12, adj_22);
    df::adj_load(var_x, var_8, adj_x, adj_8, adj_21);
    if (var_20) {
    	label0:;
    }
    df::adj_load(var_indices, var_15, adj_indices, adj_15, adj_16);
    df::adj_add(var_13, var_14, adj_13, adj_14, adj_15);
    df::adj_mul(var_1, var_4, adj_1, adj_4, adj_13);
    df::adj_load(var_indices, var_11, adj_indices, adj_11, adj_12);
    df::adj_add(var_9, var_10, adj_9, adj_10, adj_11);
    df::adj_mul(var_1, var_4, adj_1, adj_4, adj_9);
    df::adj_load(var_indices, var_7, adj_indices, adj_7, adj_8);
    df::adj_add(var_5, var_6, adj_5, adj_6, adj_7);
    df::adj_mul(var_1, var_4, adj_1, adj_4, adj_5);
    df::adj_load(var_x, var_2, adj_x, adj_2, adj_3);
    df::adj_mod(var_0, var_num_particles, adj_0, adj_num_particles, adj_2);
    df::adj_div(var_0, var_num_particles, adj_0, adj_num_particles, adj_1);
    return;

}



// Python entry points
void eval_triangles_contact_cpu_forward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_triangles_contact_cpu_kernel_forward(
			var_num_particles,
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<mat22*>(var_pose),
			cast<float*>(var_activation),
			var_k_mu,
			var_k_lambda,
			var_k_damp,
			var_k_drag,
			var_k_lift,
			cast<df::float3*>(var_f));
    }
}

void eval_triangles_contact_cpu_backward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f, 
	int adj_num_particles,
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	float adj_k_mu,
	float adj_k_lambda,
	float adj_k_damp,
	float adj_k_drag,
	float adj_k_lift,
	torch::Tensor adj_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_triangles_contact_cpu_kernel_backward(
			var_num_particles,
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<mat22*>(var_pose),
			cast<float*>(var_activation),
			var_k_mu,
			var_k_lambda,
			var_k_damp,
			var_k_drag,
			var_k_lift,
			cast<df::float3*>(var_f), 
			adj_num_particles,
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<int*>(adj_indices),
			cast<mat22*>(adj_pose),
			cast<float*>(adj_activation),
			adj_k_mu,
			adj_k_lambda,
			adj_k_damp,
			adj_k_drag,
			adj_k_lift,
			cast<df::float3*>(adj_f));
    }
}



// Python entry points
void eval_triangles_contact_cpu_forward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f);

void eval_triangles_contact_cpu_backward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	float var_k_mu,
	float var_k_lambda,
	float var_k_damp,
	float var_k_drag,
	float var_k_lift,
	torch::Tensor var_f, 
	int adj_num_particles,
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	float adj_k_mu,
	float adj_k_lambda,
	float adj_k_damp,
	float adj_k_drag,
	float adj_k_lift,
	torch::Tensor adj_f);


void eval_triangles_rigid_contacts_cpu_kernel_forward(
	int var_num_particles,
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	df::float3* var_rigid_x,
	quat* var_rigid_r,
	df::float3* var_rigid_v,
	df::float3* var_rigid_w,
	int* var_contact_body,
	df::float3* var_contact_point,
	float* var_contact_dist,
	int* var_contact_mat,
	float* var_materials,
	df::float3* var_tri_f)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    int var_3;
    df::float3 var_4;
    float var_5;
    int var_6;
    const int var_7 = 4;
    int var_8;
    const int var_9 = 0;
    int var_10;
    float var_11;
    int var_12;
    const int var_13 = 1;
    int var_14;
    float var_15;
    int var_16;
    const int var_17 = 2;
    int var_18;
    float var_19;
    int var_20;
    const int var_21 = 3;
    int var_22;
    float var_23;
    df::float3 var_24;
    quat var_25;
    df::float3 var_26;
    df::float3 var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    int var_36;
    int var_37;
    int var_38;
    int var_39;
    int var_40;
    int var_41;
    int var_42;
    int var_43;
    int var_44;
    df::float3 var_45;
    df::float3 var_46;
    df::float3 var_47;
    df::float3 var_48;
    df::float3 var_49;
    df::float3 var_50;
    df::float3 var_51;
    float var_52;
    df::float3 var_53;
    float var_54;
    df::float3 var_55;
    df::float3 var_56;
    float var_57;
    df::float3 var_58;
    df::float3 var_59;
    df::float3 var_60;
    float var_61;
    df::float3 var_62;
    const float var_63 = 0.05;
    float var_64;
    const float var_65 = 0.0;
    float var_66;
    float var_67;
    float var_68;
    df::float3 var_69;
    float var_70;
    df::float3 var_71;
    df::float3 var_72;
    float var_73;
    df::float3 var_74;
    df::float3 var_75;
    df::float3 var_76;
    float var_77;
    df::float3 var_78;
    df::float3 var_79;
    float var_80;
    float var_81;
    float var_82;
    float var_83;
    float var_84;
    float var_85;
    float var_86;
    float var_87;
    const float var_88 = 1.0;
    df::float3 var_89;
    df::float3 var_90;
    df::float3 var_91;
    df::float3 var_92;
    df::float3 var_93;
    float var_94;
    float var_95;
    df::float3 var_96;
    float var_97;
    float var_98;
    df::float3 var_99;
    df::float3 var_100;
    df::float3 var_101;
    float var_102;
    float var_103;
    df::float3 var_104;
    float var_105;
    df::float3 var_106;
    df::float3 var_107;
    float var_108;
    df::float3 var_109;
    float var_110;
    df::float3 var_111;
    float var_112;
    df::float3 var_113;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::div(var_0, var_num_particles);
    var_2 = df::mod(var_0, var_num_particles);
    var_3 = df::load(var_contact_body, var_2);
    var_4 = df::load(var_contact_point, var_2);
    var_5 = df::load(var_contact_dist, var_2);
    var_6 = df::load(var_contact_mat, var_2);
    var_8 = df::mul(var_6, var_7);
    var_10 = df::add(var_8, var_9);
    var_11 = df::load(var_materials, var_10);
    var_12 = df::mul(var_6, var_7);
    var_14 = df::add(var_12, var_13);
    var_15 = df::load(var_materials, var_14);
    var_16 = df::mul(var_6, var_7);
    var_18 = df::add(var_16, var_17);
    var_19 = df::load(var_materials, var_18);
    var_20 = df::mul(var_6, var_7);
    var_22 = df::add(var_20, var_21);
    var_23 = df::load(var_materials, var_22);
    var_24 = df::load(var_rigid_x, var_3);
    var_25 = df::load(var_rigid_r, var_3);
    var_26 = df::load(var_rigid_v, var_3);
    var_27 = df::load(var_rigid_w, var_3);
    var_28 = df::rotate(var_25, var_4);
    var_29 = df::add(var_24, var_28);
    var_30 = df::sub(var_29, var_24);
    var_31 = df::normalize(var_30);
    var_32 = df::mul(var_31, var_5);
    var_33 = df::add(var_29, var_32);
    var_34 = df::cross(var_27, var_30);
    var_35 = df::add(var_26, var_34);
    var_36 = df::mul(var_1, var_21);
    var_37 = df::add(var_36, var_9);
    var_38 = df::load(var_indices, var_37);
    var_39 = df::mul(var_1, var_21);
    var_40 = df::add(var_39, var_13);
    var_41 = df::load(var_indices, var_40);
    var_42 = df::mul(var_1, var_21);
    var_43 = df::add(var_42, var_17);
    var_44 = df::load(var_indices, var_43);
    var_45 = df::load(var_x, var_38);
    var_46 = df::load(var_x, var_41);
    var_47 = df::load(var_x, var_44);
    var_48 = df::load(var_v, var_38);
    var_49 = df::load(var_v, var_41);
    var_50 = df::load(var_v, var_44);
    var_51 = triangle_closest_point_barycentric_cpu_func(var_45, var_46, var_47, var_33);
    var_52 = df::index(var_51, var_9);
    var_53 = df::mul(var_45, var_52);
    var_54 = df::index(var_51, var_13);
    var_55 = df::mul(var_46, var_54);
    var_56 = df::add(var_53, var_55);
    var_57 = df::index(var_51, var_17);
    var_58 = df::mul(var_47, var_57);
    var_59 = df::add(var_56, var_58);
    var_60 = df::sub(var_33, var_59);
    var_61 = df::dot(var_60, var_60);
    var_62 = df::normalize(var_60);
    var_64 = df::sub(var_61, var_63);
    var_66 = df::min(var_64, var_65);
    var_67 = df::mul(var_66, var_11);
    var_68 = df::index(var_51, var_9);
    var_69 = df::mul(var_48, var_68);
    var_70 = df::index(var_51, var_13);
    var_71 = df::mul(var_49, var_70);
    var_72 = df::add(var_69, var_71);
    var_73 = df::index(var_51, var_17);
    var_74 = df::mul(var_50, var_73);
    var_75 = df::add(var_72, var_74);
    var_76 = df::sub(var_75, var_35);
    var_77 = df::dot(var_62, var_76);
    var_78 = df::mul(var_62, var_77);
    var_79 = df::sub(var_76, var_78);
    var_80 = df::max(var_77, var_65);
    var_81 = df::mul(var_80, var_15);
    var_82 = df::step(var_66);
    var_83 = df::mul(var_81, var_82);
    var_84 = df::sub(var_65, var_83);
    var_85 = df::add(var_67, var_84);
    var_86 = df::mul(var_23, var_85);
    var_87 = df::sub(var_65, var_86);
    var_89 = df::float3(var_65, var_65, var_88);
    var_90 = df::cross(var_62, var_89);
    var_91 = df::float3(var_88, var_65, var_65);
    var_92 = df::cross(var_62, var_91);
    var_93 = df::mul(var_90, var_19);
    var_94 = df::dot(var_93, var_79);
    var_95 = df::clamp(var_94, var_86, var_87);
    var_96 = df::mul(var_92, var_19);
    var_97 = df::dot(var_96, var_79);
    var_98 = df::clamp(var_97, var_86, var_87);
    var_99 = df::mul(var_90, var_95);
    var_100 = df::mul(var_92, var_98);
    var_101 = df::add(var_99, var_100);
    var_102 = df::step(var_66);
    var_103 = df::sub(var_65, var_102);
    var_104 = df::mul(var_101, var_103);
    var_105 = df::add(var_67, var_84);
    var_106 = df::mul(var_62, var_105);
    var_107 = df::add(var_106, var_104);
    var_108 = df::index(var_51, var_9);
    var_109 = df::mul(var_107, var_108);
    df::atomic_add(var_tri_f, var_38, var_109);
    var_110 = df::index(var_51, var_13);
    var_111 = df::mul(var_107, var_110);
    df::atomic_add(var_tri_f, var_41, var_111);
    var_112 = df::index(var_51, var_17);
    var_113 = df::mul(var_107, var_112);
    df::atomic_add(var_tri_f, var_44, var_113);
    

}

void eval_triangles_rigid_contacts_cpu_kernel_backward(
	int var_num_particles,
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	df::float3* var_rigid_x,
	quat* var_rigid_r,
	df::float3* var_rigid_v,
	df::float3* var_rigid_w,
	int* var_contact_body,
	df::float3* var_contact_point,
	float* var_contact_dist,
	int* var_contact_mat,
	float* var_materials,
	df::float3* var_tri_f, 
	int adj_num_particles,
	df::float3* adj_x,
	df::float3* adj_v,
	int* adj_indices,
	df::float3* adj_rigid_x,
	quat* adj_rigid_r,
	df::float3* adj_rigid_v,
	df::float3* adj_rigid_w,
	int* adj_contact_body,
	df::float3* adj_contact_point,
	float* adj_contact_dist,
	int* adj_contact_mat,
	float* adj_materials,
	df::float3* adj_tri_f)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    int var_3;
    df::float3 var_4;
    float var_5;
    int var_6;
    const int var_7 = 4;
    int var_8;
    const int var_9 = 0;
    int var_10;
    float var_11;
    int var_12;
    const int var_13 = 1;
    int var_14;
    float var_15;
    int var_16;
    const int var_17 = 2;
    int var_18;
    float var_19;
    int var_20;
    const int var_21 = 3;
    int var_22;
    float var_23;
    df::float3 var_24;
    quat var_25;
    df::float3 var_26;
    df::float3 var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    int var_36;
    int var_37;
    int var_38;
    int var_39;
    int var_40;
    int var_41;
    int var_42;
    int var_43;
    int var_44;
    df::float3 var_45;
    df::float3 var_46;
    df::float3 var_47;
    df::float3 var_48;
    df::float3 var_49;
    df::float3 var_50;
    df::float3 var_51;
    float var_52;
    df::float3 var_53;
    float var_54;
    df::float3 var_55;
    df::float3 var_56;
    float var_57;
    df::float3 var_58;
    df::float3 var_59;
    df::float3 var_60;
    float var_61;
    df::float3 var_62;
    const float var_63 = 0.05;
    float var_64;
    const float var_65 = 0.0;
    float var_66;
    float var_67;
    float var_68;
    df::float3 var_69;
    float var_70;
    df::float3 var_71;
    df::float3 var_72;
    float var_73;
    df::float3 var_74;
    df::float3 var_75;
    df::float3 var_76;
    float var_77;
    df::float3 var_78;
    df::float3 var_79;
    float var_80;
    float var_81;
    float var_82;
    float var_83;
    float var_84;
    float var_85;
    float var_86;
    float var_87;
    const float var_88 = 1.0;
    df::float3 var_89;
    df::float3 var_90;
    df::float3 var_91;
    df::float3 var_92;
    df::float3 var_93;
    float var_94;
    float var_95;
    df::float3 var_96;
    float var_97;
    float var_98;
    df::float3 var_99;
    df::float3 var_100;
    df::float3 var_101;
    float var_102;
    float var_103;
    df::float3 var_104;
    float var_105;
    df::float3 var_106;
    df::float3 var_107;
    float var_108;
    df::float3 var_109;
    float var_110;
    df::float3 var_111;
    float var_112;
    df::float3 var_113;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    df::float3 adj_4 = 0;
    float adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    int adj_10 = 0;
    float adj_11 = 0;
    int adj_12 = 0;
    int adj_13 = 0;
    int adj_14 = 0;
    float adj_15 = 0;
    int adj_16 = 0;
    int adj_17 = 0;
    int adj_18 = 0;
    float adj_19 = 0;
    int adj_20 = 0;
    int adj_21 = 0;
    int adj_22 = 0;
    float adj_23 = 0;
    df::float3 adj_24 = 0;
    quat adj_25 = 0;
    df::float3 adj_26 = 0;
    df::float3 adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    df::float3 adj_32 = 0;
    df::float3 adj_33 = 0;
    df::float3 adj_34 = 0;
    df::float3 adj_35 = 0;
    int adj_36 = 0;
    int adj_37 = 0;
    int adj_38 = 0;
    int adj_39 = 0;
    int adj_40 = 0;
    int adj_41 = 0;
    int adj_42 = 0;
    int adj_43 = 0;
    int adj_44 = 0;
    df::float3 adj_45 = 0;
    df::float3 adj_46 = 0;
    df::float3 adj_47 = 0;
    df::float3 adj_48 = 0;
    df::float3 adj_49 = 0;
    df::float3 adj_50 = 0;
    df::float3 adj_51 = 0;
    float adj_52 = 0;
    df::float3 adj_53 = 0;
    float adj_54 = 0;
    df::float3 adj_55 = 0;
    df::float3 adj_56 = 0;
    float adj_57 = 0;
    df::float3 adj_58 = 0;
    df::float3 adj_59 = 0;
    df::float3 adj_60 = 0;
    float adj_61 = 0;
    df::float3 adj_62 = 0;
    float adj_63 = 0;
    float adj_64 = 0;
    float adj_65 = 0;
    float adj_66 = 0;
    float adj_67 = 0;
    float adj_68 = 0;
    df::float3 adj_69 = 0;
    float adj_70 = 0;
    df::float3 adj_71 = 0;
    df::float3 adj_72 = 0;
    float adj_73 = 0;
    df::float3 adj_74 = 0;
    df::float3 adj_75 = 0;
    df::float3 adj_76 = 0;
    float adj_77 = 0;
    df::float3 adj_78 = 0;
    df::float3 adj_79 = 0;
    float adj_80 = 0;
    float adj_81 = 0;
    float adj_82 = 0;
    float adj_83 = 0;
    float adj_84 = 0;
    float adj_85 = 0;
    float adj_86 = 0;
    float adj_87 = 0;
    float adj_88 = 0;
    df::float3 adj_89 = 0;
    df::float3 adj_90 = 0;
    df::float3 adj_91 = 0;
    df::float3 adj_92 = 0;
    df::float3 adj_93 = 0;
    float adj_94 = 0;
    float adj_95 = 0;
    df::float3 adj_96 = 0;
    float adj_97 = 0;
    float adj_98 = 0;
    df::float3 adj_99 = 0;
    df::float3 adj_100 = 0;
    df::float3 adj_101 = 0;
    float adj_102 = 0;
    float adj_103 = 0;
    df::float3 adj_104 = 0;
    float adj_105 = 0;
    df::float3 adj_106 = 0;
    df::float3 adj_107 = 0;
    float adj_108 = 0;
    df::float3 adj_109 = 0;
    float adj_110 = 0;
    df::float3 adj_111 = 0;
    float adj_112 = 0;
    df::float3 adj_113 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::div(var_0, var_num_particles);
    var_2 = df::mod(var_0, var_num_particles);
    var_3 = df::load(var_contact_body, var_2);
    var_4 = df::load(var_contact_point, var_2);
    var_5 = df::load(var_contact_dist, var_2);
    var_6 = df::load(var_contact_mat, var_2);
    var_8 = df::mul(var_6, var_7);
    var_10 = df::add(var_8, var_9);
    var_11 = df::load(var_materials, var_10);
    var_12 = df::mul(var_6, var_7);
    var_14 = df::add(var_12, var_13);
    var_15 = df::load(var_materials, var_14);
    var_16 = df::mul(var_6, var_7);
    var_18 = df::add(var_16, var_17);
    var_19 = df::load(var_materials, var_18);
    var_20 = df::mul(var_6, var_7);
    var_22 = df::add(var_20, var_21);
    var_23 = df::load(var_materials, var_22);
    var_24 = df::load(var_rigid_x, var_3);
    var_25 = df::load(var_rigid_r, var_3);
    var_26 = df::load(var_rigid_v, var_3);
    var_27 = df::load(var_rigid_w, var_3);
    var_28 = df::rotate(var_25, var_4);
    var_29 = df::add(var_24, var_28);
    var_30 = df::sub(var_29, var_24);
    var_31 = df::normalize(var_30);
    var_32 = df::mul(var_31, var_5);
    var_33 = df::add(var_29, var_32);
    var_34 = df::cross(var_27, var_30);
    var_35 = df::add(var_26, var_34);
    var_36 = df::mul(var_1, var_21);
    var_37 = df::add(var_36, var_9);
    var_38 = df::load(var_indices, var_37);
    var_39 = df::mul(var_1, var_21);
    var_40 = df::add(var_39, var_13);
    var_41 = df::load(var_indices, var_40);
    var_42 = df::mul(var_1, var_21);
    var_43 = df::add(var_42, var_17);
    var_44 = df::load(var_indices, var_43);
    var_45 = df::load(var_x, var_38);
    var_46 = df::load(var_x, var_41);
    var_47 = df::load(var_x, var_44);
    var_48 = df::load(var_v, var_38);
    var_49 = df::load(var_v, var_41);
    var_50 = df::load(var_v, var_44);
    var_51 = triangle_closest_point_barycentric_cpu_func(var_45, var_46, var_47, var_33);
    var_52 = df::index(var_51, var_9);
    var_53 = df::mul(var_45, var_52);
    var_54 = df::index(var_51, var_13);
    var_55 = df::mul(var_46, var_54);
    var_56 = df::add(var_53, var_55);
    var_57 = df::index(var_51, var_17);
    var_58 = df::mul(var_47, var_57);
    var_59 = df::add(var_56, var_58);
    var_60 = df::sub(var_33, var_59);
    var_61 = df::dot(var_60, var_60);
    var_62 = df::normalize(var_60);
    var_64 = df::sub(var_61, var_63);
    var_66 = df::min(var_64, var_65);
    var_67 = df::mul(var_66, var_11);
    var_68 = df::index(var_51, var_9);
    var_69 = df::mul(var_48, var_68);
    var_70 = df::index(var_51, var_13);
    var_71 = df::mul(var_49, var_70);
    var_72 = df::add(var_69, var_71);
    var_73 = df::index(var_51, var_17);
    var_74 = df::mul(var_50, var_73);
    var_75 = df::add(var_72, var_74);
    var_76 = df::sub(var_75, var_35);
    var_77 = df::dot(var_62, var_76);
    var_78 = df::mul(var_62, var_77);
    var_79 = df::sub(var_76, var_78);
    var_80 = df::max(var_77, var_65);
    var_81 = df::mul(var_80, var_15);
    var_82 = df::step(var_66);
    var_83 = df::mul(var_81, var_82);
    var_84 = df::sub(var_65, var_83);
    var_85 = df::add(var_67, var_84);
    var_86 = df::mul(var_23, var_85);
    var_87 = df::sub(var_65, var_86);
    var_89 = df::float3(var_65, var_65, var_88);
    var_90 = df::cross(var_62, var_89);
    var_91 = df::float3(var_88, var_65, var_65);
    var_92 = df::cross(var_62, var_91);
    var_93 = df::mul(var_90, var_19);
    var_94 = df::dot(var_93, var_79);
    var_95 = df::clamp(var_94, var_86, var_87);
    var_96 = df::mul(var_92, var_19);
    var_97 = df::dot(var_96, var_79);
    var_98 = df::clamp(var_97, var_86, var_87);
    var_99 = df::mul(var_90, var_95);
    var_100 = df::mul(var_92, var_98);
    var_101 = df::add(var_99, var_100);
    var_102 = df::step(var_66);
    var_103 = df::sub(var_65, var_102);
    var_104 = df::mul(var_101, var_103);
    var_105 = df::add(var_67, var_84);
    var_106 = df::mul(var_62, var_105);
    var_107 = df::add(var_106, var_104);
    var_108 = df::index(var_51, var_9);
    var_109 = df::mul(var_107, var_108);
    df::atomic_add(var_tri_f, var_38, var_109);
    var_110 = df::index(var_51, var_13);
    var_111 = df::mul(var_107, var_110);
    df::atomic_add(var_tri_f, var_41, var_111);
    var_112 = df::index(var_51, var_17);
    var_113 = df::mul(var_107, var_112);
    df::atomic_add(var_tri_f, var_44, var_113);
    //---------
    // reverse
    df::adj_atomic_add(var_tri_f, var_44, var_113, adj_tri_f, adj_44, adj_113);
    df::adj_mul(var_107, var_112, adj_107, adj_112, adj_113);
    df::adj_index(var_51, var_17, adj_51, adj_17, adj_112);
    df::adj_atomic_add(var_tri_f, var_41, var_111, adj_tri_f, adj_41, adj_111);
    df::adj_mul(var_107, var_110, adj_107, adj_110, adj_111);
    df::adj_index(var_51, var_13, adj_51, adj_13, adj_110);
    df::adj_atomic_add(var_tri_f, var_38, var_109, adj_tri_f, adj_38, adj_109);
    df::adj_mul(var_107, var_108, adj_107, adj_108, adj_109);
    df::adj_index(var_51, var_9, adj_51, adj_9, adj_108);
    df::adj_add(var_106, var_104, adj_106, adj_104, adj_107);
    df::adj_mul(var_62, var_105, adj_62, adj_105, adj_106);
    df::adj_add(var_67, var_84, adj_67, adj_84, adj_105);
    df::adj_mul(var_101, var_103, adj_101, adj_103, adj_104);
    df::adj_sub(var_65, var_102, adj_65, adj_102, adj_103);
    df::adj_step(var_66, adj_66, adj_102);
    df::adj_add(var_99, var_100, adj_99, adj_100, adj_101);
    df::adj_mul(var_92, var_98, adj_92, adj_98, adj_100);
    df::adj_mul(var_90, var_95, adj_90, adj_95, adj_99);
    df::adj_clamp(var_97, var_86, var_87, adj_97, adj_86, adj_87, adj_98);
    df::adj_dot(var_96, var_79, adj_96, adj_79, adj_97);
    df::adj_mul(var_92, var_19, adj_92, adj_19, adj_96);
    df::adj_clamp(var_94, var_86, var_87, adj_94, adj_86, adj_87, adj_95);
    df::adj_dot(var_93, var_79, adj_93, adj_79, adj_94);
    df::adj_mul(var_90, var_19, adj_90, adj_19, adj_93);
    df::adj_cross(var_62, var_91, adj_62, adj_91, adj_92);
    df::adj_float3(var_88, var_65, var_65, adj_88, adj_65, adj_65, adj_91);
    df::adj_cross(var_62, var_89, adj_62, adj_89, adj_90);
    df::adj_float3(var_65, var_65, var_88, adj_65, adj_65, adj_88, adj_89);
    df::adj_sub(var_65, var_86, adj_65, adj_86, adj_87);
    df::adj_mul(var_23, var_85, adj_23, adj_85, adj_86);
    df::adj_add(var_67, var_84, adj_67, adj_84, adj_85);
    df::adj_sub(var_65, var_83, adj_65, adj_83, adj_84);
    df::adj_mul(var_81, var_82, adj_81, adj_82, adj_83);
    df::adj_step(var_66, adj_66, adj_82);
    df::adj_mul(var_80, var_15, adj_80, adj_15, adj_81);
    df::adj_max(var_77, var_65, adj_77, adj_65, adj_80);
    df::adj_sub(var_76, var_78, adj_76, adj_78, adj_79);
    df::adj_mul(var_62, var_77, adj_62, adj_77, adj_78);
    df::adj_dot(var_62, var_76, adj_62, adj_76, adj_77);
    df::adj_sub(var_75, var_35, adj_75, adj_35, adj_76);
    df::adj_add(var_72, var_74, adj_72, adj_74, adj_75);
    df::adj_mul(var_50, var_73, adj_50, adj_73, adj_74);
    df::adj_index(var_51, var_17, adj_51, adj_17, adj_73);
    df::adj_add(var_69, var_71, adj_69, adj_71, adj_72);
    df::adj_mul(var_49, var_70, adj_49, adj_70, adj_71);
    df::adj_index(var_51, var_13, adj_51, adj_13, adj_70);
    df::adj_mul(var_48, var_68, adj_48, adj_68, adj_69);
    df::adj_index(var_51, var_9, adj_51, adj_9, adj_68);
    df::adj_mul(var_66, var_11, adj_66, adj_11, adj_67);
    df::adj_min(var_64, var_65, adj_64, adj_65, adj_66);
    df::adj_sub(var_61, var_63, adj_61, adj_63, adj_64);
    df::adj_normalize(var_60, adj_60, adj_62);
    df::adj_dot(var_60, var_60, adj_60, adj_60, adj_61);
    df::adj_sub(var_33, var_59, adj_33, adj_59, adj_60);
    df::adj_add(var_56, var_58, adj_56, adj_58, adj_59);
    df::adj_mul(var_47, var_57, adj_47, adj_57, adj_58);
    df::adj_index(var_51, var_17, adj_51, adj_17, adj_57);
    df::adj_add(var_53, var_55, adj_53, adj_55, adj_56);
    df::adj_mul(var_46, var_54, adj_46, adj_54, adj_55);
    df::adj_index(var_51, var_13, adj_51, adj_13, adj_54);
    df::adj_mul(var_45, var_52, adj_45, adj_52, adj_53);
    df::adj_index(var_51, var_9, adj_51, adj_9, adj_52);
    adj_triangle_closest_point_barycentric_cpu_func(var_45, var_46, var_47, var_33, adj_45, adj_46, adj_47, adj_33, adj_51);
    df::adj_load(var_v, var_44, adj_v, adj_44, adj_50);
    df::adj_load(var_v, var_41, adj_v, adj_41, adj_49);
    df::adj_load(var_v, var_38, adj_v, adj_38, adj_48);
    df::adj_load(var_x, var_44, adj_x, adj_44, adj_47);
    df::adj_load(var_x, var_41, adj_x, adj_41, adj_46);
    df::adj_load(var_x, var_38, adj_x, adj_38, adj_45);
    df::adj_load(var_indices, var_43, adj_indices, adj_43, adj_44);
    df::adj_add(var_42, var_17, adj_42, adj_17, adj_43);
    df::adj_mul(var_1, var_21, adj_1, adj_21, adj_42);
    df::adj_load(var_indices, var_40, adj_indices, adj_40, adj_41);
    df::adj_add(var_39, var_13, adj_39, adj_13, adj_40);
    df::adj_mul(var_1, var_21, adj_1, adj_21, adj_39);
    df::adj_load(var_indices, var_37, adj_indices, adj_37, adj_38);
    df::adj_add(var_36, var_9, adj_36, adj_9, adj_37);
    df::adj_mul(var_1, var_21, adj_1, adj_21, adj_36);
    df::adj_add(var_26, var_34, adj_26, adj_34, adj_35);
    df::adj_cross(var_27, var_30, adj_27, adj_30, adj_34);
    df::adj_add(var_29, var_32, adj_29, adj_32, adj_33);
    df::adj_mul(var_31, var_5, adj_31, adj_5, adj_32);
    df::adj_normalize(var_30, adj_30, adj_31);
    df::adj_sub(var_29, var_24, adj_29, adj_24, adj_30);
    df::adj_add(var_24, var_28, adj_24, adj_28, adj_29);
    df::adj_rotate(var_25, var_4, adj_25, adj_4, adj_28);
    df::adj_load(var_rigid_w, var_3, adj_rigid_w, adj_3, adj_27);
    df::adj_load(var_rigid_v, var_3, adj_rigid_v, adj_3, adj_26);
    df::adj_load(var_rigid_r, var_3, adj_rigid_r, adj_3, adj_25);
    df::adj_load(var_rigid_x, var_3, adj_rigid_x, adj_3, adj_24);
    df::adj_load(var_materials, var_22, adj_materials, adj_22, adj_23);
    df::adj_add(var_20, var_21, adj_20, adj_21, adj_22);
    df::adj_mul(var_6, var_7, adj_6, adj_7, adj_20);
    df::adj_load(var_materials, var_18, adj_materials, adj_18, adj_19);
    df::adj_add(var_16, var_17, adj_16, adj_17, adj_18);
    df::adj_mul(var_6, var_7, adj_6, adj_7, adj_16);
    df::adj_load(var_materials, var_14, adj_materials, adj_14, adj_15);
    df::adj_add(var_12, var_13, adj_12, adj_13, adj_14);
    df::adj_mul(var_6, var_7, adj_6, adj_7, adj_12);
    df::adj_load(var_materials, var_10, adj_materials, adj_10, adj_11);
    df::adj_add(var_8, var_9, adj_8, adj_9, adj_10);
    df::adj_mul(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_load(var_contact_mat, var_2, adj_contact_mat, adj_2, adj_6);
    df::adj_load(var_contact_dist, var_2, adj_contact_dist, adj_2, adj_5);
    df::adj_load(var_contact_point, var_2, adj_contact_point, adj_2, adj_4);
    df::adj_load(var_contact_body, var_2, adj_contact_body, adj_2, adj_3);
    df::adj_mod(var_0, var_num_particles, adj_0, adj_num_particles, adj_2);
    df::adj_div(var_0, var_num_particles, adj_0, adj_num_particles, adj_1);
    return;

}



// Python entry points
void eval_triangles_rigid_contacts_cpu_forward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_tri_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_triangles_rigid_contacts_cpu_kernel_forward(
			var_num_particles,
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<df::float3*>(var_rigid_x),
			cast<quat*>(var_rigid_r),
			cast<df::float3*>(var_rigid_v),
			cast<df::float3*>(var_rigid_w),
			cast<int*>(var_contact_body),
			cast<df::float3*>(var_contact_point),
			cast<float*>(var_contact_dist),
			cast<int*>(var_contact_mat),
			cast<float*>(var_materials),
			cast<df::float3*>(var_tri_f));
    }
}

void eval_triangles_rigid_contacts_cpu_backward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_tri_f, 
	int adj_num_particles,
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_rigid_x,
	torch::Tensor adj_rigid_r,
	torch::Tensor adj_rigid_v,
	torch::Tensor adj_rigid_w,
	torch::Tensor adj_contact_body,
	torch::Tensor adj_contact_point,
	torch::Tensor adj_contact_dist,
	torch::Tensor adj_contact_mat,
	torch::Tensor adj_materials,
	torch::Tensor adj_tri_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_triangles_rigid_contacts_cpu_kernel_backward(
			var_num_particles,
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<df::float3*>(var_rigid_x),
			cast<quat*>(var_rigid_r),
			cast<df::float3*>(var_rigid_v),
			cast<df::float3*>(var_rigid_w),
			cast<int*>(var_contact_body),
			cast<df::float3*>(var_contact_point),
			cast<float*>(var_contact_dist),
			cast<int*>(var_contact_mat),
			cast<float*>(var_materials),
			cast<df::float3*>(var_tri_f), 
			adj_num_particles,
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<int*>(adj_indices),
			cast<df::float3*>(adj_rigid_x),
			cast<quat*>(adj_rigid_r),
			cast<df::float3*>(adj_rigid_v),
			cast<df::float3*>(adj_rigid_w),
			cast<int*>(adj_contact_body),
			cast<df::float3*>(adj_contact_point),
			cast<float*>(adj_contact_dist),
			cast<int*>(adj_contact_mat),
			cast<float*>(adj_materials),
			cast<df::float3*>(adj_tri_f));
    }
}



// Python entry points
void eval_triangles_rigid_contacts_cpu_forward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_tri_f);

void eval_triangles_rigid_contacts_cpu_backward(int dim, 
	int var_num_particles,
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_tri_f, 
	int adj_num_particles,
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_rigid_x,
	torch::Tensor adj_rigid_r,
	torch::Tensor adj_rigid_v,
	torch::Tensor adj_rigid_w,
	torch::Tensor adj_contact_body,
	torch::Tensor adj_contact_point,
	torch::Tensor adj_contact_dist,
	torch::Tensor adj_contact_mat,
	torch::Tensor adj_materials,
	torch::Tensor adj_tri_f);


void eval_bending_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	float* var_rest,
	float var_ke,
	float var_kd,
	df::float3* var_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 4;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    int var_14;
    const int var_15 = 3;
    int var_16;
    int var_17;
    float var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    df::float3 var_22;
    df::float3 var_23;
    df::float3 var_24;
    df::float3 var_25;
    df::float3 var_26;
    df::float3 var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    float var_33;
    float var_34;
    const float var_35 = 1.0;
    float var_36;
    float var_37;
    float var_38;
    float var_39;
    float var_40;
    df::float3 var_41;
    df::float3 var_42;
    df::float3 var_43;
    df::float3 var_44;
    df::float3 var_45;
    df::float3 var_46;
    float var_47;
    df::float3 var_48;
    float var_49;
    float var_50;
    float var_51;
    float var_52;
    df::float3 var_53;
    df::float3 var_54;
    df::float3 var_55;
    float var_56;
    df::float3 var_57;
    df::float3 var_58;
    float var_59;
    df::float3 var_60;
    df::float3 var_61;
    df::float3 var_62;
    float var_63;
    df::float3 var_64;
    df::float3 var_65;
    float var_66;
    df::float3 var_67;
    df::float3 var_68;
    float var_69;
    float var_70;
    float var_71;
    float var_72;
    float var_73;
    float var_74;
    float var_75;
    float var_76;
    float var_77;
    float var_78;
    const float var_79 = 0.0;
    float var_80;
    float var_81;
    float var_82;
    df::float3 var_83;
    df::float3 var_84;
    df::float3 var_85;
    df::float3 var_86;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::mul(var_0, var_1);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_indices, var_16);
    var_18 = df::load(var_rest, var_0);
    var_19 = df::load(var_x, var_5);
    var_20 = df::load(var_x, var_9);
    var_21 = df::load(var_x, var_13);
    var_22 = df::load(var_x, var_17);
    var_23 = df::load(var_v, var_5);
    var_24 = df::load(var_v, var_9);
    var_25 = df::load(var_v, var_13);
    var_26 = df::load(var_v, var_17);
    var_27 = df::sub(var_21, var_19);
    var_28 = df::sub(var_22, var_19);
    var_29 = df::cross(var_27, var_28);
    var_30 = df::sub(var_22, var_20);
    var_31 = df::sub(var_21, var_20);
    var_32 = df::cross(var_30, var_31);
    var_33 = df::length(var_29);
    var_34 = df::length(var_32);
    var_36 = df::div(var_35, var_33);
    var_37 = df::div(var_35, var_34);
    var_38 = df::dot(var_29, var_32);
    var_39 = df::mul(var_38, var_36);
    var_40 = df::mul(var_39, var_37);
    var_41 = df::mul(var_29, var_36);
    var_42 = df::mul(var_41, var_36);
    var_43 = df::mul(var_32, var_37);
    var_44 = df::mul(var_43, var_37);
    var_45 = df::sub(var_22, var_21);
    var_46 = df::normalize(var_45);
    var_47 = df::length(var_45);
    var_48 = df::cross(var_44, var_42);
    var_49 = df::dot(var_48, var_46);
    var_50 = df::sign(var_49);
    var_51 = df::acos(var_40);
    var_52 = df::mul(var_51, var_50);
    var_53 = df::mul(var_42, var_47);
    var_54 = df::mul(var_44, var_47);
    var_55 = df::sub(var_19, var_22);
    var_56 = df::dot(var_55, var_46);
    var_57 = df::mul(var_42, var_56);
    var_58 = df::sub(var_20, var_22);
    var_59 = df::dot(var_58, var_46);
    var_60 = df::mul(var_44, var_59);
    var_61 = df::add(var_57, var_60);
    var_62 = df::sub(var_21, var_19);
    var_63 = df::dot(var_62, var_46);
    var_64 = df::mul(var_42, var_63);
    var_65 = df::sub(var_21, var_20);
    var_66 = df::dot(var_65, var_46);
    var_67 = df::mul(var_44, var_66);
    var_68 = df::add(var_64, var_67);
    var_69 = df::sub(var_52, var_18);
    var_70 = df::mul(var_ke, var_69);
    var_71 = df::dot(var_53, var_23);
    var_72 = df::dot(var_54, var_24);
    var_73 = df::add(var_71, var_72);
    var_74 = df::dot(var_61, var_25);
    var_75 = df::add(var_73, var_74);
    var_76 = df::dot(var_68, var_26);
    var_77 = df::add(var_75, var_76);
    var_78 = df::mul(var_kd, var_77);
    var_80 = df::add(var_70, var_78);
    var_81 = df::mul(var_47, var_80);
    var_82 = df::sub(var_79, var_81);
    var_83 = df::mul(var_53, var_82);
    df::atomic_add(var_f, var_5, var_83);
    var_84 = df::mul(var_54, var_82);
    df::atomic_add(var_f, var_9, var_84);
    var_85 = df::mul(var_61, var_82);
    df::atomic_add(var_f, var_13, var_85);
    var_86 = df::mul(var_68, var_82);
    df::atomic_add(var_f, var_17, var_86);
    

}

void eval_bending_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	float* var_rest,
	float var_ke,
	float var_kd,
	df::float3* var_f, 
	df::float3* adj_x,
	df::float3* adj_v,
	int* adj_indices,
	float* adj_rest,
	float adj_ke,
	float adj_kd,
	df::float3* adj_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 4;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    int var_14;
    const int var_15 = 3;
    int var_16;
    int var_17;
    float var_18;
    df::float3 var_19;
    df::float3 var_20;
    df::float3 var_21;
    df::float3 var_22;
    df::float3 var_23;
    df::float3 var_24;
    df::float3 var_25;
    df::float3 var_26;
    df::float3 var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    float var_33;
    float var_34;
    const float var_35 = 1.0;
    float var_36;
    float var_37;
    float var_38;
    float var_39;
    float var_40;
    df::float3 var_41;
    df::float3 var_42;
    df::float3 var_43;
    df::float3 var_44;
    df::float3 var_45;
    df::float3 var_46;
    float var_47;
    df::float3 var_48;
    float var_49;
    float var_50;
    float var_51;
    float var_52;
    df::float3 var_53;
    df::float3 var_54;
    df::float3 var_55;
    float var_56;
    df::float3 var_57;
    df::float3 var_58;
    float var_59;
    df::float3 var_60;
    df::float3 var_61;
    df::float3 var_62;
    float var_63;
    df::float3 var_64;
    df::float3 var_65;
    float var_66;
    df::float3 var_67;
    df::float3 var_68;
    float var_69;
    float var_70;
    float var_71;
    float var_72;
    float var_73;
    float var_74;
    float var_75;
    float var_76;
    float var_77;
    float var_78;
    const float var_79 = 0.0;
    float var_80;
    float var_81;
    float var_82;
    df::float3 var_83;
    df::float3 var_84;
    df::float3 var_85;
    df::float3 var_86;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    int adj_10 = 0;
    int adj_11 = 0;
    int adj_12 = 0;
    int adj_13 = 0;
    int adj_14 = 0;
    int adj_15 = 0;
    int adj_16 = 0;
    int adj_17 = 0;
    float adj_18 = 0;
    df::float3 adj_19 = 0;
    df::float3 adj_20 = 0;
    df::float3 adj_21 = 0;
    df::float3 adj_22 = 0;
    df::float3 adj_23 = 0;
    df::float3 adj_24 = 0;
    df::float3 adj_25 = 0;
    df::float3 adj_26 = 0;
    df::float3 adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    df::float3 adj_32 = 0;
    float adj_33 = 0;
    float adj_34 = 0;
    float adj_35 = 0;
    float adj_36 = 0;
    float adj_37 = 0;
    float adj_38 = 0;
    float adj_39 = 0;
    float adj_40 = 0;
    df::float3 adj_41 = 0;
    df::float3 adj_42 = 0;
    df::float3 adj_43 = 0;
    df::float3 adj_44 = 0;
    df::float3 adj_45 = 0;
    df::float3 adj_46 = 0;
    float adj_47 = 0;
    df::float3 adj_48 = 0;
    float adj_49 = 0;
    float adj_50 = 0;
    float adj_51 = 0;
    float adj_52 = 0;
    df::float3 adj_53 = 0;
    df::float3 adj_54 = 0;
    df::float3 adj_55 = 0;
    float adj_56 = 0;
    df::float3 adj_57 = 0;
    df::float3 adj_58 = 0;
    float adj_59 = 0;
    df::float3 adj_60 = 0;
    df::float3 adj_61 = 0;
    df::float3 adj_62 = 0;
    float adj_63 = 0;
    df::float3 adj_64 = 0;
    df::float3 adj_65 = 0;
    float adj_66 = 0;
    df::float3 adj_67 = 0;
    df::float3 adj_68 = 0;
    float adj_69 = 0;
    float adj_70 = 0;
    float adj_71 = 0;
    float adj_72 = 0;
    float adj_73 = 0;
    float adj_74 = 0;
    float adj_75 = 0;
    float adj_76 = 0;
    float adj_77 = 0;
    float adj_78 = 0;
    float adj_79 = 0;
    float adj_80 = 0;
    float adj_81 = 0;
    float adj_82 = 0;
    df::float3 adj_83 = 0;
    df::float3 adj_84 = 0;
    df::float3 adj_85 = 0;
    df::float3 adj_86 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::mul(var_0, var_1);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_indices, var_16);
    var_18 = df::load(var_rest, var_0);
    var_19 = df::load(var_x, var_5);
    var_20 = df::load(var_x, var_9);
    var_21 = df::load(var_x, var_13);
    var_22 = df::load(var_x, var_17);
    var_23 = df::load(var_v, var_5);
    var_24 = df::load(var_v, var_9);
    var_25 = df::load(var_v, var_13);
    var_26 = df::load(var_v, var_17);
    var_27 = df::sub(var_21, var_19);
    var_28 = df::sub(var_22, var_19);
    var_29 = df::cross(var_27, var_28);
    var_30 = df::sub(var_22, var_20);
    var_31 = df::sub(var_21, var_20);
    var_32 = df::cross(var_30, var_31);
    var_33 = df::length(var_29);
    var_34 = df::length(var_32);
    var_36 = df::div(var_35, var_33);
    var_37 = df::div(var_35, var_34);
    var_38 = df::dot(var_29, var_32);
    var_39 = df::mul(var_38, var_36);
    var_40 = df::mul(var_39, var_37);
    var_41 = df::mul(var_29, var_36);
    var_42 = df::mul(var_41, var_36);
    var_43 = df::mul(var_32, var_37);
    var_44 = df::mul(var_43, var_37);
    var_45 = df::sub(var_22, var_21);
    var_46 = df::normalize(var_45);
    var_47 = df::length(var_45);
    var_48 = df::cross(var_44, var_42);
    var_49 = df::dot(var_48, var_46);
    var_50 = df::sign(var_49);
    var_51 = df::acos(var_40);
    var_52 = df::mul(var_51, var_50);
    var_53 = df::mul(var_42, var_47);
    var_54 = df::mul(var_44, var_47);
    var_55 = df::sub(var_19, var_22);
    var_56 = df::dot(var_55, var_46);
    var_57 = df::mul(var_42, var_56);
    var_58 = df::sub(var_20, var_22);
    var_59 = df::dot(var_58, var_46);
    var_60 = df::mul(var_44, var_59);
    var_61 = df::add(var_57, var_60);
    var_62 = df::sub(var_21, var_19);
    var_63 = df::dot(var_62, var_46);
    var_64 = df::mul(var_42, var_63);
    var_65 = df::sub(var_21, var_20);
    var_66 = df::dot(var_65, var_46);
    var_67 = df::mul(var_44, var_66);
    var_68 = df::add(var_64, var_67);
    var_69 = df::sub(var_52, var_18);
    var_70 = df::mul(var_ke, var_69);
    var_71 = df::dot(var_53, var_23);
    var_72 = df::dot(var_54, var_24);
    var_73 = df::add(var_71, var_72);
    var_74 = df::dot(var_61, var_25);
    var_75 = df::add(var_73, var_74);
    var_76 = df::dot(var_68, var_26);
    var_77 = df::add(var_75, var_76);
    var_78 = df::mul(var_kd, var_77);
    var_80 = df::add(var_70, var_78);
    var_81 = df::mul(var_47, var_80);
    var_82 = df::sub(var_79, var_81);
    var_83 = df::mul(var_53, var_82);
    df::atomic_add(var_f, var_5, var_83);
    var_84 = df::mul(var_54, var_82);
    df::atomic_add(var_f, var_9, var_84);
    var_85 = df::mul(var_61, var_82);
    df::atomic_add(var_f, var_13, var_85);
    var_86 = df::mul(var_68, var_82);
    df::atomic_add(var_f, var_17, var_86);
    //---------
    // reverse
    df::adj_atomic_add(var_f, var_17, var_86, adj_f, adj_17, adj_86);
    df::adj_mul(var_68, var_82, adj_68, adj_82, adj_86);
    df::adj_atomic_add(var_f, var_13, var_85, adj_f, adj_13, adj_85);
    df::adj_mul(var_61, var_82, adj_61, adj_82, adj_85);
    df::adj_atomic_add(var_f, var_9, var_84, adj_f, adj_9, adj_84);
    df::adj_mul(var_54, var_82, adj_54, adj_82, adj_84);
    df::adj_atomic_add(var_f, var_5, var_83, adj_f, adj_5, adj_83);
    df::adj_mul(var_53, var_82, adj_53, adj_82, adj_83);
    df::adj_sub(var_79, var_81, adj_79, adj_81, adj_82);
    df::adj_mul(var_47, var_80, adj_47, adj_80, adj_81);
    df::adj_add(var_70, var_78, adj_70, adj_78, adj_80);
    df::adj_mul(var_kd, var_77, adj_kd, adj_77, adj_78);
    df::adj_add(var_75, var_76, adj_75, adj_76, adj_77);
    df::adj_dot(var_68, var_26, adj_68, adj_26, adj_76);
    df::adj_add(var_73, var_74, adj_73, adj_74, adj_75);
    df::adj_dot(var_61, var_25, adj_61, adj_25, adj_74);
    df::adj_add(var_71, var_72, adj_71, adj_72, adj_73);
    df::adj_dot(var_54, var_24, adj_54, adj_24, adj_72);
    df::adj_dot(var_53, var_23, adj_53, adj_23, adj_71);
    df::adj_mul(var_ke, var_69, adj_ke, adj_69, adj_70);
    df::adj_sub(var_52, var_18, adj_52, adj_18, adj_69);
    df::adj_add(var_64, var_67, adj_64, adj_67, adj_68);
    df::adj_mul(var_44, var_66, adj_44, adj_66, adj_67);
    df::adj_dot(var_65, var_46, adj_65, adj_46, adj_66);
    df::adj_sub(var_21, var_20, adj_21, adj_20, adj_65);
    df::adj_mul(var_42, var_63, adj_42, adj_63, adj_64);
    df::adj_dot(var_62, var_46, adj_62, adj_46, adj_63);
    df::adj_sub(var_21, var_19, adj_21, adj_19, adj_62);
    df::adj_add(var_57, var_60, adj_57, adj_60, adj_61);
    df::adj_mul(var_44, var_59, adj_44, adj_59, adj_60);
    df::adj_dot(var_58, var_46, adj_58, adj_46, adj_59);
    df::adj_sub(var_20, var_22, adj_20, adj_22, adj_58);
    df::adj_mul(var_42, var_56, adj_42, adj_56, adj_57);
    df::adj_dot(var_55, var_46, adj_55, adj_46, adj_56);
    df::adj_sub(var_19, var_22, adj_19, adj_22, adj_55);
    df::adj_mul(var_44, var_47, adj_44, adj_47, adj_54);
    df::adj_mul(var_42, var_47, adj_42, adj_47, adj_53);
    df::adj_mul(var_51, var_50, adj_51, adj_50, adj_52);
    df::adj_acos(var_40, adj_40, adj_51);
    df::adj_sign(var_49, adj_49, adj_50);
    df::adj_dot(var_48, var_46, adj_48, adj_46, adj_49);
    df::adj_cross(var_44, var_42, adj_44, adj_42, adj_48);
    df::adj_length(var_45, adj_45, adj_47);
    df::adj_normalize(var_45, adj_45, adj_46);
    df::adj_sub(var_22, var_21, adj_22, adj_21, adj_45);
    df::adj_mul(var_43, var_37, adj_43, adj_37, adj_44);
    df::adj_mul(var_32, var_37, adj_32, adj_37, adj_43);
    df::adj_mul(var_41, var_36, adj_41, adj_36, adj_42);
    df::adj_mul(var_29, var_36, adj_29, adj_36, adj_41);
    df::adj_mul(var_39, var_37, adj_39, adj_37, adj_40);
    df::adj_mul(var_38, var_36, adj_38, adj_36, adj_39);
    df::adj_dot(var_29, var_32, adj_29, adj_32, adj_38);
    df::adj_div(var_35, var_34, adj_35, adj_34, adj_37);
    df::adj_div(var_35, var_33, adj_35, adj_33, adj_36);
    df::adj_length(var_32, adj_32, adj_34);
    df::adj_length(var_29, adj_29, adj_33);
    df::adj_cross(var_30, var_31, adj_30, adj_31, adj_32);
    df::adj_sub(var_21, var_20, adj_21, adj_20, adj_31);
    df::adj_sub(var_22, var_20, adj_22, adj_20, adj_30);
    df::adj_cross(var_27, var_28, adj_27, adj_28, adj_29);
    df::adj_sub(var_22, var_19, adj_22, adj_19, adj_28);
    df::adj_sub(var_21, var_19, adj_21, adj_19, adj_27);
    df::adj_load(var_v, var_17, adj_v, adj_17, adj_26);
    df::adj_load(var_v, var_13, adj_v, adj_13, adj_25);
    df::adj_load(var_v, var_9, adj_v, adj_9, adj_24);
    df::adj_load(var_v, var_5, adj_v, adj_5, adj_23);
    df::adj_load(var_x, var_17, adj_x, adj_17, adj_22);
    df::adj_load(var_x, var_13, adj_x, adj_13, adj_21);
    df::adj_load(var_x, var_9, adj_x, adj_9, adj_20);
    df::adj_load(var_x, var_5, adj_x, adj_5, adj_19);
    df::adj_load(var_rest, var_0, adj_rest, adj_0, adj_18);
    df::adj_load(var_indices, var_16, adj_indices, adj_16, adj_17);
    df::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_14);
    df::adj_load(var_indices, var_12, adj_indices, adj_12, adj_13);
    df::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_10);
    df::adj_load(var_indices, var_8, adj_indices, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_6);
    df::adj_load(var_indices, var_4, adj_indices, adj_4, adj_5);
    df::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    return;

}



// Python entry points
void eval_bending_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rest,
	float var_ke,
	float var_kd,
	torch::Tensor var_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_bending_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<float*>(var_rest),
			var_ke,
			var_kd,
			cast<df::float3*>(var_f));
    }
}

void eval_bending_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rest,
	float var_ke,
	float var_kd,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_rest,
	float adj_ke,
	float adj_kd,
	torch::Tensor adj_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_bending_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<float*>(var_rest),
			var_ke,
			var_kd,
			cast<df::float3*>(var_f), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<int*>(adj_indices),
			cast<float*>(adj_rest),
			adj_ke,
			adj_kd,
			cast<df::float3*>(adj_f));
    }
}



// Python entry points
void eval_bending_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rest,
	float var_ke,
	float var_kd,
	torch::Tensor var_f);

void eval_bending_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_rest,
	float var_ke,
	float var_kd,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_rest,
	float adj_ke,
	float adj_kd,
	torch::Tensor adj_f);


void eval_tetrahedra_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	mat33* var_pose,
	float* var_activation,
	float* var_materials,
	df::float3* var_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 4;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    int var_14;
    const int var_15 = 3;
    int var_16;
    int var_17;
    float var_18;
    int var_19;
    int var_20;
    float var_21;
    int var_22;
    int var_23;
    float var_24;
    int var_25;
    int var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    df::float3 var_36;
    df::float3 var_37;
    df::float3 var_38;
    df::float3 var_39;
    df::float3 var_40;
    df::float3 var_41;
    mat33 var_42;
    mat33 var_43;
    float var_44;
    const float var_45 = 6.0;
    float var_46;
    const float var_47 = 1.0;
    float var_48;
    float var_49;
    float var_50;
    const float var_51 = 4.0;
    float var_52;
    float var_53;
    float var_54;
    float var_55;
    float var_56;
    float var_57;
    mat33 var_58;
    mat33 var_59;
    mat33 var_60;
    float var_61;
    float var_62;
    float var_63;
    df::float3 var_64;
    float var_65;
    float var_66;
    float var_67;
    df::float3 var_68;
    float var_69;
    float var_70;
    float var_71;
    df::float3 var_72;
    float var_73;
    float var_74;
    float var_75;
    float var_76;
    float var_77;
    mat33 var_78;
    float var_79;
    float var_80;
    float var_81;
    mat33 var_82;
    mat33 var_83;
    mat33 var_84;
    mat33 var_85;
    mat33 var_86;
    float var_87;
    float var_88;
    float var_89;
    df::float3 var_90;
    float var_91;
    float var_92;
    float var_93;
    df::float3 var_94;
    float var_95;
    float var_96;
    float var_97;
    df::float3 var_98;
    float var_99;
    float var_100;
    df::float3 var_101;
    df::float3 var_102;
    df::float3 var_103;
    df::float3 var_104;
    df::float3 var_105;
    df::float3 var_106;
    float var_107;
    float var_108;
    float var_109;
    float var_110;
    float var_111;
    float var_112;
    float var_113;
    float var_114;
    float var_115;
    float var_116;
    df::float3 var_117;
    df::float3 var_118;
    df::float3 var_119;
    df::float3 var_120;
    df::float3 var_121;
    df::float3 var_122;
    df::float3 var_123;
    df::float3 var_124;
    const float var_125 = 0.0;
    float var_126;
    df::float3 var_127;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::mul(var_0, var_1);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_indices, var_16);
    var_18 = df::load(var_activation, var_0);
    var_19 = df::mul(var_0, var_15);
    var_20 = df::add(var_19, var_3);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::mul(var_0, var_15);
    var_23 = df::add(var_22, var_7);
    var_24 = df::load(var_materials, var_23);
    var_25 = df::mul(var_0, var_15);
    var_26 = df::add(var_25, var_11);
    var_27 = df::load(var_materials, var_26);
    var_28 = df::load(var_x, var_5);
    var_29 = df::load(var_x, var_9);
    var_30 = df::load(var_x, var_13);
    var_31 = df::load(var_x, var_17);
    var_32 = df::load(var_v, var_5);
    var_33 = df::load(var_v, var_9);
    var_34 = df::load(var_v, var_13);
    var_35 = df::load(var_v, var_17);
    var_36 = df::sub(var_29, var_28);
    var_37 = df::sub(var_30, var_28);
    var_38 = df::sub(var_31, var_28);
    var_39 = df::sub(var_33, var_32);
    var_40 = df::sub(var_34, var_32);
    var_41 = df::sub(var_35, var_32);
    var_42 = df::mat33(var_36, var_37, var_38);
    var_43 = df::load(var_pose, var_0);
    var_44 = df::determinant(var_43);
    var_46 = df::mul(var_44, var_45);
    var_48 = df::div(var_47, var_46);
    var_49 = df::div(var_21, var_24);
    var_50 = df::add(var_47, var_49);
    var_52 = df::mul(var_51, var_24);
    var_53 = df::div(var_21, var_52);
    var_54 = df::sub(var_50, var_53);
    var_55 = df::mul(var_21, var_48);
    var_56 = df::mul(var_24, var_48);
    var_57 = df::mul(var_27, var_48);
    var_58 = df::mul(var_42, var_43);
    var_59 = df::mat33(var_39, var_40, var_41);
    var_60 = df::mul(var_59, var_43);
    var_61 = df::index(var_58, var_3, var_3);
    var_62 = df::index(var_58, var_7, var_3);
    var_63 = df::index(var_58, var_11, var_3);
    var_64 = df::float3(var_61, var_62, var_63);
    var_65 = df::index(var_58, var_3, var_7);
    var_66 = df::index(var_58, var_7, var_7);
    var_67 = df::index(var_58, var_11, var_7);
    var_68 = df::float3(var_65, var_66, var_67);
    var_69 = df::index(var_58, var_3, var_11);
    var_70 = df::index(var_58, var_7, var_11);
    var_71 = df::index(var_58, var_11, var_11);
    var_72 = df::float3(var_69, var_70, var_71);
    var_73 = df::dot(var_64, var_64);
    var_74 = df::dot(var_68, var_68);
    var_75 = df::add(var_73, var_74);
    var_76 = df::dot(var_72, var_72);
    var_77 = df::add(var_75, var_76);
    var_78 = df::mul(var_58, var_55);
    var_79 = df::add(var_77, var_47);
    var_80 = df::div(var_47, var_79);
    var_81 = df::sub(var_47, var_80);
    var_82 = df::mul(var_78, var_81);
    var_83 = df::mul(var_60, var_57);
    var_84 = df::add(var_82, var_83);
    var_85 = df::transpose(var_43);
    var_86 = df::mul(var_84, var_85);
    var_87 = df::index(var_86, var_3, var_3);
    var_88 = df::index(var_86, var_7, var_3);
    var_89 = df::index(var_86, var_11, var_3);
    var_90 = df::float3(var_87, var_88, var_89);
    var_91 = df::index(var_86, var_3, var_7);
    var_92 = df::index(var_86, var_7, var_7);
    var_93 = df::index(var_86, var_11, var_7);
    var_94 = df::float3(var_91, var_92, var_93);
    var_95 = df::index(var_86, var_3, var_11);
    var_96 = df::index(var_86, var_7, var_11);
    var_97 = df::index(var_86, var_11, var_11);
    var_98 = df::float3(var_95, var_96, var_97);
    var_99 = df::determinant(var_58);
    var_100 = df::div(var_46, var_45);
    var_101 = df::cross(var_37, var_38);
    var_102 = df::mul(var_101, var_100);
    var_103 = df::cross(var_38, var_36);
    var_104 = df::mul(var_103, var_100);
    var_105 = df::cross(var_36, var_37);
    var_106 = df::mul(var_105, var_100);
    var_107 = df::sub(var_99, var_54);
    var_108 = df::add(var_107, var_18);
    var_109 = df::mul(var_108, var_56);
    var_110 = df::dot(var_102, var_33);
    var_111 = df::dot(var_104, var_34);
    var_112 = df::add(var_110, var_111);
    var_113 = df::dot(var_106, var_35);
    var_114 = df::add(var_112, var_113);
    var_115 = df::mul(var_114, var_57);
    var_116 = df::add(var_109, var_115);
    var_117 = df::mul(var_102, var_116);
    var_118 = df::add(var_90, var_117);
    var_119 = df::mul(var_104, var_116);
    var_120 = df::add(var_94, var_119);
    var_121 = df::mul(var_106, var_116);
    var_122 = df::add(var_98, var_121);
    var_123 = df::add(var_118, var_120);
    var_124 = df::add(var_123, var_122);
    var_126 = df::sub(var_125, var_47);
    var_127 = df::mul(var_124, var_126);
    df::atomic_sub(var_f, var_5, var_127);
    df::atomic_sub(var_f, var_9, var_118);
    df::atomic_sub(var_f, var_13, var_120);
    df::atomic_sub(var_f, var_17, var_122);
    

}

void eval_tetrahedra_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	int* var_indices,
	mat33* var_pose,
	float* var_activation,
	float* var_materials,
	df::float3* var_f, 
	df::float3* adj_x,
	df::float3* adj_v,
	int* adj_indices,
	mat33* adj_pose,
	float* adj_activation,
	float* adj_materials,
	df::float3* adj_f)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 4;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    int var_14;
    const int var_15 = 3;
    int var_16;
    int var_17;
    float var_18;
    int var_19;
    int var_20;
    float var_21;
    int var_22;
    int var_23;
    float var_24;
    int var_25;
    int var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    df::float3 var_36;
    df::float3 var_37;
    df::float3 var_38;
    df::float3 var_39;
    df::float3 var_40;
    df::float3 var_41;
    mat33 var_42;
    mat33 var_43;
    float var_44;
    const float var_45 = 6.0;
    float var_46;
    const float var_47 = 1.0;
    float var_48;
    float var_49;
    float var_50;
    const float var_51 = 4.0;
    float var_52;
    float var_53;
    float var_54;
    float var_55;
    float var_56;
    float var_57;
    mat33 var_58;
    mat33 var_59;
    mat33 var_60;
    float var_61;
    float var_62;
    float var_63;
    df::float3 var_64;
    float var_65;
    float var_66;
    float var_67;
    df::float3 var_68;
    float var_69;
    float var_70;
    float var_71;
    df::float3 var_72;
    float var_73;
    float var_74;
    float var_75;
    float var_76;
    float var_77;
    mat33 var_78;
    float var_79;
    float var_80;
    float var_81;
    mat33 var_82;
    mat33 var_83;
    mat33 var_84;
    mat33 var_85;
    mat33 var_86;
    float var_87;
    float var_88;
    float var_89;
    df::float3 var_90;
    float var_91;
    float var_92;
    float var_93;
    df::float3 var_94;
    float var_95;
    float var_96;
    float var_97;
    df::float3 var_98;
    float var_99;
    float var_100;
    df::float3 var_101;
    df::float3 var_102;
    df::float3 var_103;
    df::float3 var_104;
    df::float3 var_105;
    df::float3 var_106;
    float var_107;
    float var_108;
    float var_109;
    float var_110;
    float var_111;
    float var_112;
    float var_113;
    float var_114;
    float var_115;
    float var_116;
    df::float3 var_117;
    df::float3 var_118;
    df::float3 var_119;
    df::float3 var_120;
    df::float3 var_121;
    df::float3 var_122;
    df::float3 var_123;
    df::float3 var_124;
    const float var_125 = 0.0;
    float var_126;
    df::float3 var_127;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    int adj_10 = 0;
    int adj_11 = 0;
    int adj_12 = 0;
    int adj_13 = 0;
    int adj_14 = 0;
    int adj_15 = 0;
    int adj_16 = 0;
    int adj_17 = 0;
    float adj_18 = 0;
    int adj_19 = 0;
    int adj_20 = 0;
    float adj_21 = 0;
    int adj_22 = 0;
    int adj_23 = 0;
    float adj_24 = 0;
    int adj_25 = 0;
    int adj_26 = 0;
    float adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    df::float3 adj_32 = 0;
    df::float3 adj_33 = 0;
    df::float3 adj_34 = 0;
    df::float3 adj_35 = 0;
    df::float3 adj_36 = 0;
    df::float3 adj_37 = 0;
    df::float3 adj_38 = 0;
    df::float3 adj_39 = 0;
    df::float3 adj_40 = 0;
    df::float3 adj_41 = 0;
    mat33 adj_42 = 0;
    mat33 adj_43 = 0;
    float adj_44 = 0;
    float adj_45 = 0;
    float adj_46 = 0;
    float adj_47 = 0;
    float adj_48 = 0;
    float adj_49 = 0;
    float adj_50 = 0;
    float adj_51 = 0;
    float adj_52 = 0;
    float adj_53 = 0;
    float adj_54 = 0;
    float adj_55 = 0;
    float adj_56 = 0;
    float adj_57 = 0;
    mat33 adj_58 = 0;
    mat33 adj_59 = 0;
    mat33 adj_60 = 0;
    float adj_61 = 0;
    float adj_62 = 0;
    float adj_63 = 0;
    df::float3 adj_64 = 0;
    float adj_65 = 0;
    float adj_66 = 0;
    float adj_67 = 0;
    df::float3 adj_68 = 0;
    float adj_69 = 0;
    float adj_70 = 0;
    float adj_71 = 0;
    df::float3 adj_72 = 0;
    float adj_73 = 0;
    float adj_74 = 0;
    float adj_75 = 0;
    float adj_76 = 0;
    float adj_77 = 0;
    mat33 adj_78 = 0;
    float adj_79 = 0;
    float adj_80 = 0;
    float adj_81 = 0;
    mat33 adj_82 = 0;
    mat33 adj_83 = 0;
    mat33 adj_84 = 0;
    mat33 adj_85 = 0;
    mat33 adj_86 = 0;
    float adj_87 = 0;
    float adj_88 = 0;
    float adj_89 = 0;
    df::float3 adj_90 = 0;
    float adj_91 = 0;
    float adj_92 = 0;
    float adj_93 = 0;
    df::float3 adj_94 = 0;
    float adj_95 = 0;
    float adj_96 = 0;
    float adj_97 = 0;
    df::float3 adj_98 = 0;
    float adj_99 = 0;
    float adj_100 = 0;
    df::float3 adj_101 = 0;
    df::float3 adj_102 = 0;
    df::float3 adj_103 = 0;
    df::float3 adj_104 = 0;
    df::float3 adj_105 = 0;
    df::float3 adj_106 = 0;
    float adj_107 = 0;
    float adj_108 = 0;
    float adj_109 = 0;
    float adj_110 = 0;
    float adj_111 = 0;
    float adj_112 = 0;
    float adj_113 = 0;
    float adj_114 = 0;
    float adj_115 = 0;
    float adj_116 = 0;
    df::float3 adj_117 = 0;
    df::float3 adj_118 = 0;
    df::float3 adj_119 = 0;
    df::float3 adj_120 = 0;
    df::float3 adj_121 = 0;
    df::float3 adj_122 = 0;
    df::float3 adj_123 = 0;
    df::float3 adj_124 = 0;
    float adj_125 = 0;
    float adj_126 = 0;
    df::float3 adj_127 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::mul(var_0, var_1);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_indices, var_16);
    var_18 = df::load(var_activation, var_0);
    var_19 = df::mul(var_0, var_15);
    var_20 = df::add(var_19, var_3);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::mul(var_0, var_15);
    var_23 = df::add(var_22, var_7);
    var_24 = df::load(var_materials, var_23);
    var_25 = df::mul(var_0, var_15);
    var_26 = df::add(var_25, var_11);
    var_27 = df::load(var_materials, var_26);
    var_28 = df::load(var_x, var_5);
    var_29 = df::load(var_x, var_9);
    var_30 = df::load(var_x, var_13);
    var_31 = df::load(var_x, var_17);
    var_32 = df::load(var_v, var_5);
    var_33 = df::load(var_v, var_9);
    var_34 = df::load(var_v, var_13);
    var_35 = df::load(var_v, var_17);
    var_36 = df::sub(var_29, var_28);
    var_37 = df::sub(var_30, var_28);
    var_38 = df::sub(var_31, var_28);
    var_39 = df::sub(var_33, var_32);
    var_40 = df::sub(var_34, var_32);
    var_41 = df::sub(var_35, var_32);
    var_42 = df::mat33(var_36, var_37, var_38);
    var_43 = df::load(var_pose, var_0);
    var_44 = df::determinant(var_43);
    var_46 = df::mul(var_44, var_45);
    var_48 = df::div(var_47, var_46);
    var_49 = df::div(var_21, var_24);
    var_50 = df::add(var_47, var_49);
    var_52 = df::mul(var_51, var_24);
    var_53 = df::div(var_21, var_52);
    var_54 = df::sub(var_50, var_53);
    var_55 = df::mul(var_21, var_48);
    var_56 = df::mul(var_24, var_48);
    var_57 = df::mul(var_27, var_48);
    var_58 = df::mul(var_42, var_43);
    var_59 = df::mat33(var_39, var_40, var_41);
    var_60 = df::mul(var_59, var_43);
    var_61 = df::index(var_58, var_3, var_3);
    var_62 = df::index(var_58, var_7, var_3);
    var_63 = df::index(var_58, var_11, var_3);
    var_64 = df::float3(var_61, var_62, var_63);
    var_65 = df::index(var_58, var_3, var_7);
    var_66 = df::index(var_58, var_7, var_7);
    var_67 = df::index(var_58, var_11, var_7);
    var_68 = df::float3(var_65, var_66, var_67);
    var_69 = df::index(var_58, var_3, var_11);
    var_70 = df::index(var_58, var_7, var_11);
    var_71 = df::index(var_58, var_11, var_11);
    var_72 = df::float3(var_69, var_70, var_71);
    var_73 = df::dot(var_64, var_64);
    var_74 = df::dot(var_68, var_68);
    var_75 = df::add(var_73, var_74);
    var_76 = df::dot(var_72, var_72);
    var_77 = df::add(var_75, var_76);
    var_78 = df::mul(var_58, var_55);
    var_79 = df::add(var_77, var_47);
    var_80 = df::div(var_47, var_79);
    var_81 = df::sub(var_47, var_80);
    var_82 = df::mul(var_78, var_81);
    var_83 = df::mul(var_60, var_57);
    var_84 = df::add(var_82, var_83);
    var_85 = df::transpose(var_43);
    var_86 = df::mul(var_84, var_85);
    var_87 = df::index(var_86, var_3, var_3);
    var_88 = df::index(var_86, var_7, var_3);
    var_89 = df::index(var_86, var_11, var_3);
    var_90 = df::float3(var_87, var_88, var_89);
    var_91 = df::index(var_86, var_3, var_7);
    var_92 = df::index(var_86, var_7, var_7);
    var_93 = df::index(var_86, var_11, var_7);
    var_94 = df::float3(var_91, var_92, var_93);
    var_95 = df::index(var_86, var_3, var_11);
    var_96 = df::index(var_86, var_7, var_11);
    var_97 = df::index(var_86, var_11, var_11);
    var_98 = df::float3(var_95, var_96, var_97);
    var_99 = df::determinant(var_58);
    var_100 = df::div(var_46, var_45);
    var_101 = df::cross(var_37, var_38);
    var_102 = df::mul(var_101, var_100);
    var_103 = df::cross(var_38, var_36);
    var_104 = df::mul(var_103, var_100);
    var_105 = df::cross(var_36, var_37);
    var_106 = df::mul(var_105, var_100);
    var_107 = df::sub(var_99, var_54);
    var_108 = df::add(var_107, var_18);
    var_109 = df::mul(var_108, var_56);
    var_110 = df::dot(var_102, var_33);
    var_111 = df::dot(var_104, var_34);
    var_112 = df::add(var_110, var_111);
    var_113 = df::dot(var_106, var_35);
    var_114 = df::add(var_112, var_113);
    var_115 = df::mul(var_114, var_57);
    var_116 = df::add(var_109, var_115);
    var_117 = df::mul(var_102, var_116);
    var_118 = df::add(var_90, var_117);
    var_119 = df::mul(var_104, var_116);
    var_120 = df::add(var_94, var_119);
    var_121 = df::mul(var_106, var_116);
    var_122 = df::add(var_98, var_121);
    var_123 = df::add(var_118, var_120);
    var_124 = df::add(var_123, var_122);
    var_126 = df::sub(var_125, var_47);
    var_127 = df::mul(var_124, var_126);
    df::atomic_sub(var_f, var_5, var_127);
    df::atomic_sub(var_f, var_9, var_118);
    df::atomic_sub(var_f, var_13, var_120);
    df::atomic_sub(var_f, var_17, var_122);
    //---------
    // reverse
    df::adj_atomic_sub(var_f, var_17, var_122, adj_f, adj_17, adj_122);
    df::adj_atomic_sub(var_f, var_13, var_120, adj_f, adj_13, adj_120);
    df::adj_atomic_sub(var_f, var_9, var_118, adj_f, adj_9, adj_118);
    df::adj_atomic_sub(var_f, var_5, var_127, adj_f, adj_5, adj_127);
    df::adj_mul(var_124, var_126, adj_124, adj_126, adj_127);
    df::adj_sub(var_125, var_47, adj_125, adj_47, adj_126);
    df::adj_add(var_123, var_122, adj_123, adj_122, adj_124);
    df::adj_add(var_118, var_120, adj_118, adj_120, adj_123);
    df::adj_add(var_98, var_121, adj_98, adj_121, adj_122);
    df::adj_mul(var_106, var_116, adj_106, adj_116, adj_121);
    df::adj_add(var_94, var_119, adj_94, adj_119, adj_120);
    df::adj_mul(var_104, var_116, adj_104, adj_116, adj_119);
    df::adj_add(var_90, var_117, adj_90, adj_117, adj_118);
    df::adj_mul(var_102, var_116, adj_102, adj_116, adj_117);
    df::adj_add(var_109, var_115, adj_109, adj_115, adj_116);
    df::adj_mul(var_114, var_57, adj_114, adj_57, adj_115);
    df::adj_add(var_112, var_113, adj_112, adj_113, adj_114);
    df::adj_dot(var_106, var_35, adj_106, adj_35, adj_113);
    df::adj_add(var_110, var_111, adj_110, adj_111, adj_112);
    df::adj_dot(var_104, var_34, adj_104, adj_34, adj_111);
    df::adj_dot(var_102, var_33, adj_102, adj_33, adj_110);
    df::adj_mul(var_108, var_56, adj_108, adj_56, adj_109);
    df::adj_add(var_107, var_18, adj_107, adj_18, adj_108);
    df::adj_sub(var_99, var_54, adj_99, adj_54, adj_107);
    df::adj_mul(var_105, var_100, adj_105, adj_100, adj_106);
    df::adj_cross(var_36, var_37, adj_36, adj_37, adj_105);
    df::adj_mul(var_103, var_100, adj_103, adj_100, adj_104);
    df::adj_cross(var_38, var_36, adj_38, adj_36, adj_103);
    df::adj_mul(var_101, var_100, adj_101, adj_100, adj_102);
    df::adj_cross(var_37, var_38, adj_37, adj_38, adj_101);
    df::adj_div(var_46, var_45, adj_46, adj_45, adj_100);
    df::adj_determinant(var_58, adj_58, adj_99);
    df::adj_float3(var_95, var_96, var_97, adj_95, adj_96, adj_97, adj_98);
    df::adj_index(var_86, var_11, var_11, adj_86, adj_11, adj_11, adj_97);
    df::adj_index(var_86, var_7, var_11, adj_86, adj_7, adj_11, adj_96);
    df::adj_index(var_86, var_3, var_11, adj_86, adj_3, adj_11, adj_95);
    df::adj_float3(var_91, var_92, var_93, adj_91, adj_92, adj_93, adj_94);
    df::adj_index(var_86, var_11, var_7, adj_86, adj_11, adj_7, adj_93);
    df::adj_index(var_86, var_7, var_7, adj_86, adj_7, adj_7, adj_92);
    df::adj_index(var_86, var_3, var_7, adj_86, adj_3, adj_7, adj_91);
    df::adj_float3(var_87, var_88, var_89, adj_87, adj_88, adj_89, adj_90);
    df::adj_index(var_86, var_11, var_3, adj_86, adj_11, adj_3, adj_89);
    df::adj_index(var_86, var_7, var_3, adj_86, adj_7, adj_3, adj_88);
    df::adj_index(var_86, var_3, var_3, adj_86, adj_3, adj_3, adj_87);
    df::adj_mul(var_84, var_85, adj_84, adj_85, adj_86);
    df::adj_transpose(var_43, adj_43, adj_85);
    df::adj_add(var_82, var_83, adj_82, adj_83, adj_84);
    df::adj_mul(var_60, var_57, adj_60, adj_57, adj_83);
    df::adj_mul(var_78, var_81, adj_78, adj_81, adj_82);
    df::adj_sub(var_47, var_80, adj_47, adj_80, adj_81);
    df::adj_div(var_47, var_79, adj_47, adj_79, adj_80);
    df::adj_add(var_77, var_47, adj_77, adj_47, adj_79);
    df::adj_mul(var_58, var_55, adj_58, adj_55, adj_78);
    df::adj_add(var_75, var_76, adj_75, adj_76, adj_77);
    df::adj_dot(var_72, var_72, adj_72, adj_72, adj_76);
    df::adj_add(var_73, var_74, adj_73, adj_74, adj_75);
    df::adj_dot(var_68, var_68, adj_68, adj_68, adj_74);
    df::adj_dot(var_64, var_64, adj_64, adj_64, adj_73);
    df::adj_float3(var_69, var_70, var_71, adj_69, adj_70, adj_71, adj_72);
    df::adj_index(var_58, var_11, var_11, adj_58, adj_11, adj_11, adj_71);
    df::adj_index(var_58, var_7, var_11, adj_58, adj_7, adj_11, adj_70);
    df::adj_index(var_58, var_3, var_11, adj_58, adj_3, adj_11, adj_69);
    df::adj_float3(var_65, var_66, var_67, adj_65, adj_66, adj_67, adj_68);
    df::adj_index(var_58, var_11, var_7, adj_58, adj_11, adj_7, adj_67);
    df::adj_index(var_58, var_7, var_7, adj_58, adj_7, adj_7, adj_66);
    df::adj_index(var_58, var_3, var_7, adj_58, adj_3, adj_7, adj_65);
    df::adj_float3(var_61, var_62, var_63, adj_61, adj_62, adj_63, adj_64);
    df::adj_index(var_58, var_11, var_3, adj_58, adj_11, adj_3, adj_63);
    df::adj_index(var_58, var_7, var_3, adj_58, adj_7, adj_3, adj_62);
    df::adj_index(var_58, var_3, var_3, adj_58, adj_3, adj_3, adj_61);
    df::adj_mul(var_59, var_43, adj_59, adj_43, adj_60);
    df::adj_mat33(var_39, var_40, var_41, adj_39, adj_40, adj_41, adj_59);
    df::adj_mul(var_42, var_43, adj_42, adj_43, adj_58);
    df::adj_mul(var_27, var_48, adj_27, adj_48, adj_57);
    df::adj_mul(var_24, var_48, adj_24, adj_48, adj_56);
    df::adj_mul(var_21, var_48, adj_21, adj_48, adj_55);
    df::adj_sub(var_50, var_53, adj_50, adj_53, adj_54);
    df::adj_div(var_21, var_52, adj_21, adj_52, adj_53);
    df::adj_mul(var_51, var_24, adj_51, adj_24, adj_52);
    df::adj_add(var_47, var_49, adj_47, adj_49, adj_50);
    df::adj_div(var_21, var_24, adj_21, adj_24, adj_49);
    df::adj_div(var_47, var_46, adj_47, adj_46, adj_48);
    df::adj_mul(var_44, var_45, adj_44, adj_45, adj_46);
    df::adj_determinant(var_43, adj_43, adj_44);
    df::adj_load(var_pose, var_0, adj_pose, adj_0, adj_43);
    df::adj_mat33(var_36, var_37, var_38, adj_36, adj_37, adj_38, adj_42);
    df::adj_sub(var_35, var_32, adj_35, adj_32, adj_41);
    df::adj_sub(var_34, var_32, adj_34, adj_32, adj_40);
    df::adj_sub(var_33, var_32, adj_33, adj_32, adj_39);
    df::adj_sub(var_31, var_28, adj_31, adj_28, adj_38);
    df::adj_sub(var_30, var_28, adj_30, adj_28, adj_37);
    df::adj_sub(var_29, var_28, adj_29, adj_28, adj_36);
    df::adj_load(var_v, var_17, adj_v, adj_17, adj_35);
    df::adj_load(var_v, var_13, adj_v, adj_13, adj_34);
    df::adj_load(var_v, var_9, adj_v, adj_9, adj_33);
    df::adj_load(var_v, var_5, adj_v, adj_5, adj_32);
    df::adj_load(var_x, var_17, adj_x, adj_17, adj_31);
    df::adj_load(var_x, var_13, adj_x, adj_13, adj_30);
    df::adj_load(var_x, var_9, adj_x, adj_9, adj_29);
    df::adj_load(var_x, var_5, adj_x, adj_5, adj_28);
    df::adj_load(var_materials, var_26, adj_materials, adj_26, adj_27);
    df::adj_add(var_25, var_11, adj_25, adj_11, adj_26);
    df::adj_mul(var_0, var_15, adj_0, adj_15, adj_25);
    df::adj_load(var_materials, var_23, adj_materials, adj_23, adj_24);
    df::adj_add(var_22, var_7, adj_22, adj_7, adj_23);
    df::adj_mul(var_0, var_15, adj_0, adj_15, adj_22);
    df::adj_load(var_materials, var_20, adj_materials, adj_20, adj_21);
    df::adj_add(var_19, var_3, adj_19, adj_3, adj_20);
    df::adj_mul(var_0, var_15, adj_0, adj_15, adj_19);
    df::adj_load(var_activation, var_0, adj_activation, adj_0, adj_18);
    df::adj_load(var_indices, var_16, adj_indices, adj_16, adj_17);
    df::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_14);
    df::adj_load(var_indices, var_12, adj_indices, adj_12, adj_13);
    df::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_10);
    df::adj_load(var_indices, var_8, adj_indices, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_6);
    df::adj_load(var_indices, var_4, adj_indices, adj_4, adj_5);
    df::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    return;

}



// Python entry points
void eval_tetrahedra_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	torch::Tensor var_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_tetrahedra_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<mat33*>(var_pose),
			cast<float*>(var_activation),
			cast<float*>(var_materials),
			cast<df::float3*>(var_f));
    }
}

void eval_tetrahedra_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	torch::Tensor adj_materials,
	torch::Tensor adj_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_tetrahedra_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<int*>(var_indices),
			cast<mat33*>(var_pose),
			cast<float*>(var_activation),
			cast<float*>(var_materials),
			cast<df::float3*>(var_f), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<int*>(adj_indices),
			cast<mat33*>(adj_pose),
			cast<float*>(adj_activation),
			cast<float*>(adj_materials),
			cast<df::float3*>(adj_f));
    }
}



// Python entry points
void eval_tetrahedra_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	torch::Tensor var_f);

void eval_tetrahedra_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	torch::Tensor adj_materials,
	torch::Tensor adj_f);


void eval_contacts_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	df::float3* var_f)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    const float var_3 = 0.0;
    const float var_4 = 1.0;
    df::float3 var_5;
    float var_6;
    const float var_7 = 0.01;
    float var_8;
    float var_9;
    float var_10;
    df::float3 var_11;
    df::float3 var_12;
    df::float3 var_13;
    df::float3 var_14;
    float var_15;
    df::float3 var_16;
    df::float3 var_17;
    float var_18;
    float var_19;
    float var_20;
    df::float3 var_21;
    float var_22;
    float var_23;
    df::float3 var_24;
    float var_25;
    float var_26;
    df::float3 var_27;
    df::float3 var_28;
    float var_29;
    df::float3 var_30;
    df::float3 var_31;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x, var_0);
    var_2 = df::load(var_v, var_0);
    var_5 = df::float3(var_3, var_4, var_3);
    var_6 = df::dot(var_5, var_1);
    var_8 = df::sub(var_6, var_7);
    var_9 = df::min(var_8, var_3);
    var_10 = df::dot(var_5, var_2);
    var_11 = df::mul(var_5, var_10);
    var_12 = df::sub(var_2, var_11);
    var_13 = df::mul(var_5, var_9);
    var_14 = df::mul(var_13, var_ke);
    var_15 = df::min(var_10, var_3);
    var_16 = df::mul(var_5, var_15);
    var_17 = df::mul(var_16, var_kd);
    var_18 = df::mul(var_mu, var_9);
    var_19 = df::mul(var_18, var_ke);
    var_20 = df::sub(var_3, var_19);
    var_21 = df::float3(var_kf, var_3, var_3);
    var_22 = df::dot(var_21, var_12);
    var_23 = df::clamp(var_22, var_19, var_20);
    var_24 = df::float3(var_3, var_3, var_kf);
    var_25 = df::dot(var_24, var_12);
    var_26 = df::clamp(var_25, var_19, var_20);
    var_27 = df::float3(var_23, var_3, var_26);
    var_28 = df::add(var_17, var_27);
    var_29 = df::step(var_9);
    var_30 = df::mul(var_28, var_29);
    var_31 = df::add(var_14, var_30);
    df::atomic_sub(var_f, var_0, var_31);
    

}

void eval_contacts_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	df::float3* var_f, 
	df::float3* adj_x,
	df::float3* adj_v,
	float adj_ke,
	float adj_kd,
	float adj_kf,
	float adj_mu,
	df::float3* adj_f)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    const float var_3 = 0.0;
    const float var_4 = 1.0;
    df::float3 var_5;
    float var_6;
    const float var_7 = 0.01;
    float var_8;
    float var_9;
    float var_10;
    df::float3 var_11;
    df::float3 var_12;
    df::float3 var_13;
    df::float3 var_14;
    float var_15;
    df::float3 var_16;
    df::float3 var_17;
    float var_18;
    float var_19;
    float var_20;
    df::float3 var_21;
    float var_22;
    float var_23;
    df::float3 var_24;
    float var_25;
    float var_26;
    df::float3 var_27;
    df::float3 var_28;
    float var_29;
    df::float3 var_30;
    df::float3 var_31;
    //---------
    // dual vars
    int adj_0 = 0;
    df::float3 adj_1 = 0;
    df::float3 adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    df::float3 adj_5 = 0;
    float adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    float adj_10 = 0;
    df::float3 adj_11 = 0;
    df::float3 adj_12 = 0;
    df::float3 adj_13 = 0;
    df::float3 adj_14 = 0;
    float adj_15 = 0;
    df::float3 adj_16 = 0;
    df::float3 adj_17 = 0;
    float adj_18 = 0;
    float adj_19 = 0;
    float adj_20 = 0;
    df::float3 adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    df::float3 adj_24 = 0;
    float adj_25 = 0;
    float adj_26 = 0;
    df::float3 adj_27 = 0;
    df::float3 adj_28 = 0;
    float adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x, var_0);
    var_2 = df::load(var_v, var_0);
    var_5 = df::float3(var_3, var_4, var_3);
    var_6 = df::dot(var_5, var_1);
    var_8 = df::sub(var_6, var_7);
    var_9 = df::min(var_8, var_3);
    var_10 = df::dot(var_5, var_2);
    var_11 = df::mul(var_5, var_10);
    var_12 = df::sub(var_2, var_11);
    var_13 = df::mul(var_5, var_9);
    var_14 = df::mul(var_13, var_ke);
    var_15 = df::min(var_10, var_3);
    var_16 = df::mul(var_5, var_15);
    var_17 = df::mul(var_16, var_kd);
    var_18 = df::mul(var_mu, var_9);
    var_19 = df::mul(var_18, var_ke);
    var_20 = df::sub(var_3, var_19);
    var_21 = df::float3(var_kf, var_3, var_3);
    var_22 = df::dot(var_21, var_12);
    var_23 = df::clamp(var_22, var_19, var_20);
    var_24 = df::float3(var_3, var_3, var_kf);
    var_25 = df::dot(var_24, var_12);
    var_26 = df::clamp(var_25, var_19, var_20);
    var_27 = df::float3(var_23, var_3, var_26);
    var_28 = df::add(var_17, var_27);
    var_29 = df::step(var_9);
    var_30 = df::mul(var_28, var_29);
    var_31 = df::add(var_14, var_30);
    df::atomic_sub(var_f, var_0, var_31);
    //---------
    // reverse
    df::adj_atomic_sub(var_f, var_0, var_31, adj_f, adj_0, adj_31);
    df::adj_add(var_14, var_30, adj_14, adj_30, adj_31);
    df::adj_mul(var_28, var_29, adj_28, adj_29, adj_30);
    df::adj_step(var_9, adj_9, adj_29);
    df::adj_add(var_17, var_27, adj_17, adj_27, adj_28);
    df::adj_float3(var_23, var_3, var_26, adj_23, adj_3, adj_26, adj_27);
    df::adj_clamp(var_25, var_19, var_20, adj_25, adj_19, adj_20, adj_26);
    df::adj_dot(var_24, var_12, adj_24, adj_12, adj_25);
    df::adj_float3(var_3, var_3, var_kf, adj_3, adj_3, adj_kf, adj_24);
    df::adj_clamp(var_22, var_19, var_20, adj_22, adj_19, adj_20, adj_23);
    df::adj_dot(var_21, var_12, adj_21, adj_12, adj_22);
    df::adj_float3(var_kf, var_3, var_3, adj_kf, adj_3, adj_3, adj_21);
    df::adj_sub(var_3, var_19, adj_3, adj_19, adj_20);
    df::adj_mul(var_18, var_ke, adj_18, adj_ke, adj_19);
    df::adj_mul(var_mu, var_9, adj_mu, adj_9, adj_18);
    df::adj_mul(var_16, var_kd, adj_16, adj_kd, adj_17);
    df::adj_mul(var_5, var_15, adj_5, adj_15, adj_16);
    df::adj_min(var_10, var_3, adj_10, adj_3, adj_15);
    df::adj_mul(var_13, var_ke, adj_13, adj_ke, adj_14);
    df::adj_mul(var_5, var_9, adj_5, adj_9, adj_13);
    df::adj_sub(var_2, var_11, adj_2, adj_11, adj_12);
    df::adj_mul(var_5, var_10, adj_5, adj_10, adj_11);
    df::adj_dot(var_5, var_2, adj_5, adj_2, adj_10);
    df::adj_min(var_8, var_3, adj_8, adj_3, adj_9);
    df::adj_sub(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_dot(var_5, var_1, adj_5, adj_1, adj_6);
    df::adj_float3(var_3, var_4, var_3, adj_3, adj_4, adj_3, adj_5);
    df::adj_load(var_v, var_0, adj_v, adj_0, adj_2);
    df::adj_load(var_x, var_0, adj_x, adj_0, adj_1);
    return;

}



// Python entry points
void eval_contacts_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_contacts_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			var_ke,
			var_kd,
			var_kf,
			var_mu,
			cast<df::float3*>(var_f));
    }
}

void eval_contacts_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	float adj_ke,
	float adj_kd,
	float adj_kf,
	float adj_mu,
	torch::Tensor adj_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_contacts_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			var_ke,
			var_kd,
			var_kf,
			var_mu,
			cast<df::float3*>(var_f), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			adj_ke,
			adj_kd,
			adj_kf,
			adj_mu,
			cast<df::float3*>(adj_f));
    }
}



// Python entry points
void eval_contacts_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_f);

void eval_contacts_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_f, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	float adj_ke,
	float adj_kd,
	float adj_kf,
	float adj_mu,
	torch::Tensor adj_f);


void eval_soft_contacts_cpu_kernel_forward(
	int var_num_particles,
	df::float3* var_particle_x,
	df::float3* var_particle_v,
	spatial_transform* var_body_X_sc,
	spatial_vector* var_body_v_sc,
	spatial_transform* var_shape_X_co,
	int* var_shape_body,
	int* var_shape_geo_type,
	int* var_shape_geo_src,
	df::float3* var_shape_geo_scale,
	float* var_shape_materials,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	df::float3* var_particle_f,
	spatial_vector* var_body_f)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    int var_3;
    df::float3 var_4;
    df::float3 var_5;
    spatial_transform var_6;
    const int var_7 = 0;
    bool var_8;
    spatial_transform var_9;
    spatial_transform var_10;
    spatial_transform var_11;
    spatial_transform var_12;
    spatial_transform var_13;
    df::float3 var_14;
    int var_15;
    df::float3 var_16;
    const float var_17 = 0.01;
    const float var_18 = 0.0;
    df::float3 var_19;
    bool var_20;
    df::float3 var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    df::float3 var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    float var_30;
    df::float3 var_31;
    const int var_32 = 1;
    bool var_33;
    float var_34;
    float var_35;
    float var_36;
    df::float3 var_37;
    df::float3 var_38;
    float var_39;
    df::float3 var_40;
    const int var_41 = 2;
    bool var_42;
    float var_43;
    float var_44;
    float var_45;
    float var_46;
    float var_47;
    float var_48;
    float var_49;
    df::float3 var_50;
    df::float3 var_51;
    float var_52;
    df::float3 var_53;
    spatial_vector var_54;
    bool var_55;
    spatial_vector var_56;
    spatial_vector var_57;
    df::float3 var_58;
    df::float3 var_59;
    df::float3 var_60;
    df::float3 var_61;
    df::float3 var_62;
    float var_63;
    df::float3 var_64;
    df::float3 var_65;
    df::float3 var_66;
    df::float3 var_67;
    float var_68;
    df::float3 var_69;
    df::float3 var_70;
    float var_71;
    float var_72;
    float var_73;
    df::float3 var_74;
    float var_75;
    float var_76;
    df::float3 var_77;
    float var_78;
    float var_79;
    df::float3 var_80;
    df::float3 var_81;
    float var_82;
    df::float3 var_83;
    df::float3 var_84;
    df::float3 var_85;
    bool var_86;
    spatial_vector var_87;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::div(var_0, var_num_particles);
    var_2 = df::mod(var_0, var_num_particles);
    var_3 = df::load(var_shape_body, var_1);
    var_4 = df::load(var_particle_x, var_2);
    var_5 = df::load(var_particle_v, var_2);
    var_6 = df::spatial_transform_identity();
    var_8 = (var_3 >= var_7);
    if (var_8) {
    	var_9 = df::load(var_body_X_sc, var_3);
    }
    var_10 = df::select(var_8, var_6, var_9);
    var_11 = df::load(var_shape_X_co, var_1);
    var_12 = df::spatial_transform_multiply(var_10, var_11);
    var_13 = spatial_transform_inverse_cpu_func(var_12);
    var_14 = df::spatial_transform_point(var_13, var_4);
    var_15 = df::load(var_shape_geo_type, var_1);
    var_16 = df::load(var_shape_geo_scale, var_1);
    var_19 = df::float3(var_18, var_18, var_18);
    var_20 = (var_15 == var_7);
    if (var_20) {
    	var_21 = df::float3(var_18, var_18, var_18);
    	var_22 = df::index(var_16, var_7);
    	var_23 = sphere_sdf_cpu_func(var_21, var_22, var_14);
    	var_24 = df::sub(var_23, var_17);
    	var_25 = df::min(var_24, var_18);
    	var_26 = df::float3(var_18, var_18, var_18);
    	var_27 = df::index(var_16, var_7);
    	var_28 = sphere_sdf_grad_cpu_func(var_26, var_27, var_14);
    	var_29 = df::spatial_transform_vector(var_12, var_28);
    }
    var_30 = df::select(var_20, var_18, var_25);
    var_31 = df::select(var_20, var_19, var_29);
    var_33 = (var_15 == var_32);
    if (var_33) {
    	var_34 = box_sdf_cpu_func(var_16, var_14);
    	var_35 = df::sub(var_34, var_17);
    	var_36 = df::min(var_35, var_18);
    	var_37 = box_sdf_grad_cpu_func(var_16, var_14);
    	var_38 = df::spatial_transform_vector(var_12, var_37);
    }
    var_39 = df::select(var_33, var_30, var_36);
    var_40 = df::select(var_33, var_31, var_38);
    var_42 = (var_15 == var_41);
    if (var_42) {
    	var_43 = df::index(var_16, var_7);
    	var_44 = df::index(var_16, var_32);
    	var_45 = capsule_sdf_cpu_func(var_43, var_44, var_14);
    	var_46 = df::sub(var_45, var_17);
    	var_47 = df::min(var_46, var_18);
    	var_48 = df::index(var_16, var_7);
    	var_49 = df::index(var_16, var_32);
    	var_50 = capsule_sdf_grad_cpu_func(var_48, var_49, var_14);
    	var_51 = df::spatial_transform_vector(var_12, var_50);
    }
    var_52 = df::select(var_42, var_39, var_47);
    var_53 = df::select(var_42, var_40, var_51);
    var_54 = df::spatial_vector();
    var_55 = (var_3 >= var_7);
    if (var_55) {
    	var_56 = df::load(var_body_v_sc, var_3);
    }
    var_57 = df::select(var_55, var_54, var_56);
    var_58 = df::spatial_top(var_57);
    var_59 = df::spatial_bottom(var_57);
    var_60 = df::cross(var_58, var_4);
    var_61 = df::add(var_59, var_60);
    var_62 = df::sub(var_5, var_61);
    var_63 = df::dot(var_53, var_62);
    var_64 = df::mul(var_53, var_63);
    var_65 = df::sub(var_62, var_64);
    var_66 = df::mul(var_53, var_52);
    var_67 = df::mul(var_66, var_ke);
    var_68 = df::min(var_63, var_18);
    var_69 = df::mul(var_53, var_68);
    var_70 = df::mul(var_69, var_kd);
    var_71 = df::mul(var_mu, var_52);
    var_72 = df::mul(var_71, var_ke);
    var_73 = df::sub(var_18, var_72);
    var_74 = df::float3(var_kf, var_18, var_18);
    var_75 = df::dot(var_74, var_65);
    var_76 = df::clamp(var_75, var_72, var_73);
    var_77 = df::float3(var_18, var_18, var_kf);
    var_78 = df::dot(var_77, var_65);
    var_79 = df::clamp(var_78, var_72, var_73);
    var_80 = df::float3(var_76, var_18, var_79);
    var_81 = df::add(var_70, var_80);
    var_82 = df::step(var_52);
    var_83 = df::mul(var_81, var_82);
    var_84 = df::add(var_67, var_83);
    var_85 = df::cross(var_4, var_84);
    df::atomic_sub(var_particle_f, var_2, var_84);
    var_86 = (var_3 >= var_7);
    if (var_86) {
    	var_87 = df::spatial_vector(var_85, var_84);
    	df::atomic_sub(var_body_f, var_3, var_87);
    }
    

}

void eval_soft_contacts_cpu_kernel_backward(
	int var_num_particles,
	df::float3* var_particle_x,
	df::float3* var_particle_v,
	spatial_transform* var_body_X_sc,
	spatial_vector* var_body_v_sc,
	spatial_transform* var_shape_X_co,
	int* var_shape_body,
	int* var_shape_geo_type,
	int* var_shape_geo_src,
	df::float3* var_shape_geo_scale,
	float* var_shape_materials,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	df::float3* var_particle_f,
	spatial_vector* var_body_f, 
	int adj_num_particles,
	df::float3* adj_particle_x,
	df::float3* adj_particle_v,
	spatial_transform* adj_body_X_sc,
	spatial_vector* adj_body_v_sc,
	spatial_transform* adj_shape_X_co,
	int* adj_shape_body,
	int* adj_shape_geo_type,
	int* adj_shape_geo_src,
	df::float3* adj_shape_geo_scale,
	float* adj_shape_materials,
	float adj_ke,
	float adj_kd,
	float adj_kf,
	float adj_mu,
	df::float3* adj_particle_f,
	spatial_vector* adj_body_f)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    int var_3;
    df::float3 var_4;
    df::float3 var_5;
    spatial_transform var_6;
    const int var_7 = 0;
    bool var_8;
    spatial_transform var_9;
    spatial_transform var_10;
    spatial_transform var_11;
    spatial_transform var_12;
    spatial_transform var_13;
    df::float3 var_14;
    int var_15;
    df::float3 var_16;
    const float var_17 = 0.01;
    const float var_18 = 0.0;
    df::float3 var_19;
    bool var_20;
    df::float3 var_21;
    float var_22;
    float var_23;
    float var_24;
    float var_25;
    df::float3 var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    float var_30;
    df::float3 var_31;
    const int var_32 = 1;
    bool var_33;
    float var_34;
    float var_35;
    float var_36;
    df::float3 var_37;
    df::float3 var_38;
    float var_39;
    df::float3 var_40;
    const int var_41 = 2;
    bool var_42;
    float var_43;
    float var_44;
    float var_45;
    float var_46;
    float var_47;
    float var_48;
    float var_49;
    df::float3 var_50;
    df::float3 var_51;
    float var_52;
    df::float3 var_53;
    spatial_vector var_54;
    bool var_55;
    spatial_vector var_56;
    spatial_vector var_57;
    df::float3 var_58;
    df::float3 var_59;
    df::float3 var_60;
    df::float3 var_61;
    df::float3 var_62;
    float var_63;
    df::float3 var_64;
    df::float3 var_65;
    df::float3 var_66;
    df::float3 var_67;
    float var_68;
    df::float3 var_69;
    df::float3 var_70;
    float var_71;
    float var_72;
    float var_73;
    df::float3 var_74;
    float var_75;
    float var_76;
    df::float3 var_77;
    float var_78;
    float var_79;
    df::float3 var_80;
    df::float3 var_81;
    float var_82;
    df::float3 var_83;
    df::float3 var_84;
    df::float3 var_85;
    bool var_86;
    spatial_vector var_87;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    df::float3 adj_4 = 0;
    df::float3 adj_5 = 0;
    spatial_transform adj_6 = 0;
    int adj_7 = 0;
    bool adj_8 = 0;
    spatial_transform adj_9 = 0;
    spatial_transform adj_10 = 0;
    spatial_transform adj_11 = 0;
    spatial_transform adj_12 = 0;
    spatial_transform adj_13 = 0;
    df::float3 adj_14 = 0;
    int adj_15 = 0;
    df::float3 adj_16 = 0;
    float adj_17 = 0;
    float adj_18 = 0;
    df::float3 adj_19 = 0;
    bool adj_20 = 0;
    df::float3 adj_21 = 0;
    float adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    df::float3 adj_26 = 0;
    float adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    float adj_30 = 0;
    df::float3 adj_31 = 0;
    int adj_32 = 0;
    bool adj_33 = 0;
    float adj_34 = 0;
    float adj_35 = 0;
    float adj_36 = 0;
    df::float3 adj_37 = 0;
    df::float3 adj_38 = 0;
    float adj_39 = 0;
    df::float3 adj_40 = 0;
    int adj_41 = 0;
    bool adj_42 = 0;
    float adj_43 = 0;
    float adj_44 = 0;
    float adj_45 = 0;
    float adj_46 = 0;
    float adj_47 = 0;
    float adj_48 = 0;
    float adj_49 = 0;
    df::float3 adj_50 = 0;
    df::float3 adj_51 = 0;
    float adj_52 = 0;
    df::float3 adj_53 = 0;
    spatial_vector adj_54 = 0;
    bool adj_55 = 0;
    spatial_vector adj_56 = 0;
    spatial_vector adj_57 = 0;
    df::float3 adj_58 = 0;
    df::float3 adj_59 = 0;
    df::float3 adj_60 = 0;
    df::float3 adj_61 = 0;
    df::float3 adj_62 = 0;
    float adj_63 = 0;
    df::float3 adj_64 = 0;
    df::float3 adj_65 = 0;
    df::float3 adj_66 = 0;
    df::float3 adj_67 = 0;
    float adj_68 = 0;
    df::float3 adj_69 = 0;
    df::float3 adj_70 = 0;
    float adj_71 = 0;
    float adj_72 = 0;
    float adj_73 = 0;
    df::float3 adj_74 = 0;
    float adj_75 = 0;
    float adj_76 = 0;
    df::float3 adj_77 = 0;
    float adj_78 = 0;
    float adj_79 = 0;
    df::float3 adj_80 = 0;
    df::float3 adj_81 = 0;
    float adj_82 = 0;
    df::float3 adj_83 = 0;
    df::float3 adj_84 = 0;
    df::float3 adj_85 = 0;
    bool adj_86 = 0;
    spatial_vector adj_87 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::div(var_0, var_num_particles);
    var_2 = df::mod(var_0, var_num_particles);
    var_3 = df::load(var_shape_body, var_1);
    var_4 = df::load(var_particle_x, var_2);
    var_5 = df::load(var_particle_v, var_2);
    var_6 = df::spatial_transform_identity();
    var_8 = (var_3 >= var_7);
    if (var_8) {
    	var_9 = df::load(var_body_X_sc, var_3);
    }
    var_10 = df::select(var_8, var_6, var_9);
    var_11 = df::load(var_shape_X_co, var_1);
    var_12 = df::spatial_transform_multiply(var_10, var_11);
    var_13 = spatial_transform_inverse_cpu_func(var_12);
    var_14 = df::spatial_transform_point(var_13, var_4);
    var_15 = df::load(var_shape_geo_type, var_1);
    var_16 = df::load(var_shape_geo_scale, var_1);
    var_19 = df::float3(var_18, var_18, var_18);
    var_20 = (var_15 == var_7);
    if (var_20) {
    	var_21 = df::float3(var_18, var_18, var_18);
    	var_22 = df::index(var_16, var_7);
    	var_23 = sphere_sdf_cpu_func(var_21, var_22, var_14);
    	var_24 = df::sub(var_23, var_17);
    	var_25 = df::min(var_24, var_18);
    	var_26 = df::float3(var_18, var_18, var_18);
    	var_27 = df::index(var_16, var_7);
    	var_28 = sphere_sdf_grad_cpu_func(var_26, var_27, var_14);
    	var_29 = df::spatial_transform_vector(var_12, var_28);
    }
    var_30 = df::select(var_20, var_18, var_25);
    var_31 = df::select(var_20, var_19, var_29);
    var_33 = (var_15 == var_32);
    if (var_33) {
    	var_34 = box_sdf_cpu_func(var_16, var_14);
    	var_35 = df::sub(var_34, var_17);
    	var_36 = df::min(var_35, var_18);
    	var_37 = box_sdf_grad_cpu_func(var_16, var_14);
    	var_38 = df::spatial_transform_vector(var_12, var_37);
    }
    var_39 = df::select(var_33, var_30, var_36);
    var_40 = df::select(var_33, var_31, var_38);
    var_42 = (var_15 == var_41);
    if (var_42) {
    	var_43 = df::index(var_16, var_7);
    	var_44 = df::index(var_16, var_32);
    	var_45 = capsule_sdf_cpu_func(var_43, var_44, var_14);
    	var_46 = df::sub(var_45, var_17);
    	var_47 = df::min(var_46, var_18);
    	var_48 = df::index(var_16, var_7);
    	var_49 = df::index(var_16, var_32);
    	var_50 = capsule_sdf_grad_cpu_func(var_48, var_49, var_14);
    	var_51 = df::spatial_transform_vector(var_12, var_50);
    }
    var_52 = df::select(var_42, var_39, var_47);
    var_53 = df::select(var_42, var_40, var_51);
    var_54 = df::spatial_vector();
    var_55 = (var_3 >= var_7);
    if (var_55) {
    	var_56 = df::load(var_body_v_sc, var_3);
    }
    var_57 = df::select(var_55, var_54, var_56);
    var_58 = df::spatial_top(var_57);
    var_59 = df::spatial_bottom(var_57);
    var_60 = df::cross(var_58, var_4);
    var_61 = df::add(var_59, var_60);
    var_62 = df::sub(var_5, var_61);
    var_63 = df::dot(var_53, var_62);
    var_64 = df::mul(var_53, var_63);
    var_65 = df::sub(var_62, var_64);
    var_66 = df::mul(var_53, var_52);
    var_67 = df::mul(var_66, var_ke);
    var_68 = df::min(var_63, var_18);
    var_69 = df::mul(var_53, var_68);
    var_70 = df::mul(var_69, var_kd);
    var_71 = df::mul(var_mu, var_52);
    var_72 = df::mul(var_71, var_ke);
    var_73 = df::sub(var_18, var_72);
    var_74 = df::float3(var_kf, var_18, var_18);
    var_75 = df::dot(var_74, var_65);
    var_76 = df::clamp(var_75, var_72, var_73);
    var_77 = df::float3(var_18, var_18, var_kf);
    var_78 = df::dot(var_77, var_65);
    var_79 = df::clamp(var_78, var_72, var_73);
    var_80 = df::float3(var_76, var_18, var_79);
    var_81 = df::add(var_70, var_80);
    var_82 = df::step(var_52);
    var_83 = df::mul(var_81, var_82);
    var_84 = df::add(var_67, var_83);
    var_85 = df::cross(var_4, var_84);
    df::atomic_sub(var_particle_f, var_2, var_84);
    var_86 = (var_3 >= var_7);
    if (var_86) {
    	var_87 = df::spatial_vector(var_85, var_84);
    	df::atomic_sub(var_body_f, var_3, var_87);
    }
    //---------
    // reverse
    if (var_86) {
    	df::adj_atomic_sub(var_body_f, var_3, var_87, adj_body_f, adj_3, adj_87);
    	df::adj_spatial_vector(var_85, var_84, adj_85, adj_84, adj_87);
    }
    df::adj_atomic_sub(var_particle_f, var_2, var_84, adj_particle_f, adj_2, adj_84);
    df::adj_cross(var_4, var_84, adj_4, adj_84, adj_85);
    df::adj_add(var_67, var_83, adj_67, adj_83, adj_84);
    df::adj_mul(var_81, var_82, adj_81, adj_82, adj_83);
    df::adj_step(var_52, adj_52, adj_82);
    df::adj_add(var_70, var_80, adj_70, adj_80, adj_81);
    df::adj_float3(var_76, var_18, var_79, adj_76, adj_18, adj_79, adj_80);
    df::adj_clamp(var_78, var_72, var_73, adj_78, adj_72, adj_73, adj_79);
    df::adj_dot(var_77, var_65, adj_77, adj_65, adj_78);
    df::adj_float3(var_18, var_18, var_kf, adj_18, adj_18, adj_kf, adj_77);
    df::adj_clamp(var_75, var_72, var_73, adj_75, adj_72, adj_73, adj_76);
    df::adj_dot(var_74, var_65, adj_74, adj_65, adj_75);
    df::adj_float3(var_kf, var_18, var_18, adj_kf, adj_18, adj_18, adj_74);
    df::adj_sub(var_18, var_72, adj_18, adj_72, adj_73);
    df::adj_mul(var_71, var_ke, adj_71, adj_ke, adj_72);
    df::adj_mul(var_mu, var_52, adj_mu, adj_52, adj_71);
    df::adj_mul(var_69, var_kd, adj_69, adj_kd, adj_70);
    df::adj_mul(var_53, var_68, adj_53, adj_68, adj_69);
    df::adj_min(var_63, var_18, adj_63, adj_18, adj_68);
    df::adj_mul(var_66, var_ke, adj_66, adj_ke, adj_67);
    df::adj_mul(var_53, var_52, adj_53, adj_52, adj_66);
    df::adj_sub(var_62, var_64, adj_62, adj_64, adj_65);
    df::adj_mul(var_53, var_63, adj_53, adj_63, adj_64);
    df::adj_dot(var_53, var_62, adj_53, adj_62, adj_63);
    df::adj_sub(var_5, var_61, adj_5, adj_61, adj_62);
    df::adj_add(var_59, var_60, adj_59, adj_60, adj_61);
    df::adj_cross(var_58, var_4, adj_58, adj_4, adj_60);
    df::adj_spatial_bottom(var_57, adj_57, adj_59);
    df::adj_spatial_top(var_57, adj_57, adj_58);
    df::adj_select(var_55, var_54, var_56, adj_55, adj_54, adj_56, adj_57);
    if (var_55) {
    	df::adj_load(var_body_v_sc, var_3, adj_body_v_sc, adj_3, adj_56);
    }
    df::adj_select(var_42, var_40, var_51, adj_42, adj_40, adj_51, adj_53);
    df::adj_select(var_42, var_39, var_47, adj_42, adj_39, adj_47, adj_52);
    if (var_42) {
    	df::adj_spatial_transform_vector(var_12, var_50, adj_12, adj_50, adj_51);
    	adj_capsule_sdf_grad_cpu_func(var_48, var_49, var_14, adj_48, adj_49, adj_14, adj_50);
    	df::adj_index(var_16, var_32, adj_16, adj_32, adj_49);
    	df::adj_index(var_16, var_7, adj_16, adj_7, adj_48);
    	df::adj_min(var_46, var_18, adj_46, adj_18, adj_47);
    	df::adj_sub(var_45, var_17, adj_45, adj_17, adj_46);
    	adj_capsule_sdf_cpu_func(var_43, var_44, var_14, adj_43, adj_44, adj_14, adj_45);
    	df::adj_index(var_16, var_32, adj_16, adj_32, adj_44);
    	df::adj_index(var_16, var_7, adj_16, adj_7, adj_43);
    }
    df::adj_select(var_33, var_31, var_38, adj_33, adj_31, adj_38, adj_40);
    df::adj_select(var_33, var_30, var_36, adj_33, adj_30, adj_36, adj_39);
    if (var_33) {
    	df::adj_spatial_transform_vector(var_12, var_37, adj_12, adj_37, adj_38);
    	adj_box_sdf_grad_cpu_func(var_16, var_14, adj_16, adj_14, adj_37);
    	df::adj_min(var_35, var_18, adj_35, adj_18, adj_36);
    	df::adj_sub(var_34, var_17, adj_34, adj_17, adj_35);
    	adj_box_sdf_cpu_func(var_16, var_14, adj_16, adj_14, adj_34);
    }
    df::adj_select(var_20, var_19, var_29, adj_20, adj_19, adj_29, adj_31);
    df::adj_select(var_20, var_18, var_25, adj_20, adj_18, adj_25, adj_30);
    if (var_20) {
    	df::adj_spatial_transform_vector(var_12, var_28, adj_12, adj_28, adj_29);
    	adj_sphere_sdf_grad_cpu_func(var_26, var_27, var_14, adj_26, adj_27, adj_14, adj_28);
    	df::adj_index(var_16, var_7, adj_16, adj_7, adj_27);
    	df::adj_float3(var_18, var_18, var_18, adj_18, adj_18, adj_18, adj_26);
    	df::adj_min(var_24, var_18, adj_24, adj_18, adj_25);
    	df::adj_sub(var_23, var_17, adj_23, adj_17, adj_24);
    	adj_sphere_sdf_cpu_func(var_21, var_22, var_14, adj_21, adj_22, adj_14, adj_23);
    	df::adj_index(var_16, var_7, adj_16, adj_7, adj_22);
    	df::adj_float3(var_18, var_18, var_18, adj_18, adj_18, adj_18, adj_21);
    }
    df::adj_float3(var_18, var_18, var_18, adj_18, adj_18, adj_18, adj_19);
    df::adj_load(var_shape_geo_scale, var_1, adj_shape_geo_scale, adj_1, adj_16);
    df::adj_load(var_shape_geo_type, var_1, adj_shape_geo_type, adj_1, adj_15);
    df::adj_spatial_transform_point(var_13, var_4, adj_13, adj_4, adj_14);
    adj_spatial_transform_inverse_cpu_func(var_12, adj_12, adj_13);
    df::adj_spatial_transform_multiply(var_10, var_11, adj_10, adj_11, adj_12);
    df::adj_load(var_shape_X_co, var_1, adj_shape_X_co, adj_1, adj_11);
    df::adj_select(var_8, var_6, var_9, adj_8, adj_6, adj_9, adj_10);
    if (var_8) {
    	df::adj_load(var_body_X_sc, var_3, adj_body_X_sc, adj_3, adj_9);
    }
    df::adj_load(var_particle_v, var_2, adj_particle_v, adj_2, adj_5);
    df::adj_load(var_particle_x, var_2, adj_particle_x, adj_2, adj_4);
    df::adj_load(var_shape_body, var_1, adj_shape_body, adj_1, adj_3);
    df::adj_mod(var_0, var_num_particles, adj_0, adj_num_particles, adj_2);
    df::adj_div(var_0, var_num_particles, adj_0, adj_num_particles, adj_1);
    return;

}



// Python entry points
void eval_soft_contacts_cpu_forward(int dim, 
	int var_num_particles,
	torch::Tensor var_particle_x,
	torch::Tensor var_particle_v,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_v_sc,
	torch::Tensor var_shape_X_co,
	torch::Tensor var_shape_body,
	torch::Tensor var_shape_geo_type,
	torch::Tensor var_shape_geo_src,
	torch::Tensor var_shape_geo_scale,
	torch::Tensor var_shape_materials,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_particle_f,
	torch::Tensor var_body_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_soft_contacts_cpu_kernel_forward(
			var_num_particles,
			cast<df::float3*>(var_particle_x),
			cast<df::float3*>(var_particle_v),
			cast<spatial_transform*>(var_body_X_sc),
			cast<spatial_vector*>(var_body_v_sc),
			cast<spatial_transform*>(var_shape_X_co),
			cast<int*>(var_shape_body),
			cast<int*>(var_shape_geo_type),
			cast<int*>(var_shape_geo_src),
			cast<df::float3*>(var_shape_geo_scale),
			cast<float*>(var_shape_materials),
			var_ke,
			var_kd,
			var_kf,
			var_mu,
			cast<df::float3*>(var_particle_f),
			cast<spatial_vector*>(var_body_f));
    }
}

void eval_soft_contacts_cpu_backward(int dim, 
	int var_num_particles,
	torch::Tensor var_particle_x,
	torch::Tensor var_particle_v,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_v_sc,
	torch::Tensor var_shape_X_co,
	torch::Tensor var_shape_body,
	torch::Tensor var_shape_geo_type,
	torch::Tensor var_shape_geo_src,
	torch::Tensor var_shape_geo_scale,
	torch::Tensor var_shape_materials,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_particle_f,
	torch::Tensor var_body_f, 
	int adj_num_particles,
	torch::Tensor adj_particle_x,
	torch::Tensor adj_particle_v,
	torch::Tensor adj_body_X_sc,
	torch::Tensor adj_body_v_sc,
	torch::Tensor adj_shape_X_co,
	torch::Tensor adj_shape_body,
	torch::Tensor adj_shape_geo_type,
	torch::Tensor adj_shape_geo_src,
	torch::Tensor adj_shape_geo_scale,
	torch::Tensor adj_shape_materials,
	float adj_ke,
	float adj_kd,
	float adj_kf,
	float adj_mu,
	torch::Tensor adj_particle_f,
	torch::Tensor adj_body_f)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_soft_contacts_cpu_kernel_backward(
			var_num_particles,
			cast<df::float3*>(var_particle_x),
			cast<df::float3*>(var_particle_v),
			cast<spatial_transform*>(var_body_X_sc),
			cast<spatial_vector*>(var_body_v_sc),
			cast<spatial_transform*>(var_shape_X_co),
			cast<int*>(var_shape_body),
			cast<int*>(var_shape_geo_type),
			cast<int*>(var_shape_geo_src),
			cast<df::float3*>(var_shape_geo_scale),
			cast<float*>(var_shape_materials),
			var_ke,
			var_kd,
			var_kf,
			var_mu,
			cast<df::float3*>(var_particle_f),
			cast<spatial_vector*>(var_body_f), 
			adj_num_particles,
			cast<df::float3*>(adj_particle_x),
			cast<df::float3*>(adj_particle_v),
			cast<spatial_transform*>(adj_body_X_sc),
			cast<spatial_vector*>(adj_body_v_sc),
			cast<spatial_transform*>(adj_shape_X_co),
			cast<int*>(adj_shape_body),
			cast<int*>(adj_shape_geo_type),
			cast<int*>(adj_shape_geo_src),
			cast<df::float3*>(adj_shape_geo_scale),
			cast<float*>(adj_shape_materials),
			adj_ke,
			adj_kd,
			adj_kf,
			adj_mu,
			cast<df::float3*>(adj_particle_f),
			cast<spatial_vector*>(adj_body_f));
    }
}



// Python entry points
void eval_soft_contacts_cpu_forward(int dim, 
	int var_num_particles,
	torch::Tensor var_particle_x,
	torch::Tensor var_particle_v,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_v_sc,
	torch::Tensor var_shape_X_co,
	torch::Tensor var_shape_body,
	torch::Tensor var_shape_geo_type,
	torch::Tensor var_shape_geo_src,
	torch::Tensor var_shape_geo_scale,
	torch::Tensor var_shape_materials,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_particle_f,
	torch::Tensor var_body_f);

void eval_soft_contacts_cpu_backward(int dim, 
	int var_num_particles,
	torch::Tensor var_particle_x,
	torch::Tensor var_particle_v,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_v_sc,
	torch::Tensor var_shape_X_co,
	torch::Tensor var_shape_body,
	torch::Tensor var_shape_geo_type,
	torch::Tensor var_shape_geo_src,
	torch::Tensor var_shape_geo_scale,
	torch::Tensor var_shape_materials,
	float var_ke,
	float var_kd,
	float var_kf,
	float var_mu,
	torch::Tensor var_particle_f,
	torch::Tensor var_body_f, 
	int adj_num_particles,
	torch::Tensor adj_particle_x,
	torch::Tensor adj_particle_v,
	torch::Tensor adj_body_X_sc,
	torch::Tensor adj_body_v_sc,
	torch::Tensor adj_shape_X_co,
	torch::Tensor adj_shape_body,
	torch::Tensor adj_shape_geo_type,
	torch::Tensor adj_shape_geo_src,
	torch::Tensor adj_shape_geo_scale,
	torch::Tensor adj_shape_materials,
	float adj_ke,
	float adj_kd,
	float adj_kf,
	float adj_mu,
	torch::Tensor adj_particle_f,
	torch::Tensor adj_body_f);


void eval_rigid_contacts_cpu_kernel_forward(
	df::float3* var_rigid_x,
	quat* var_rigid_r,
	df::float3* var_rigid_v,
	df::float3* var_rigid_w,
	int* var_contact_body,
	df::float3* var_contact_point,
	float* var_contact_dist,
	int* var_contact_mat,
	float* var_materials,
	df::float3* var_rigid_f,
	df::float3* var_rigid_t)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    df::float3 var_2;
    float var_3;
    int var_4;
    const int var_5 = 4;
    int var_6;
    const int var_7 = 0;
    int var_8;
    float var_9;
    int var_10;
    const int var_11 = 1;
    int var_12;
    float var_13;
    int var_14;
    const int var_15 = 2;
    int var_16;
    float var_17;
    int var_18;
    const int var_19 = 3;
    int var_20;
    float var_21;
    df::float3 var_22;
    quat var_23;
    df::float3 var_24;
    df::float3 var_25;
    const float var_26 = 0.0;
    const float var_27 = 1.0;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    float var_36;
    float var_37;
    float var_38;
    df::float3 var_39;
    df::float3 var_40;
    float var_41;
    float var_42;
    float var_43;
    float var_44;
    float var_45;
    float var_46;
    float var_47;
    float var_48;
    df::float3 var_49;
    float var_50;
    float var_51;
    df::float3 var_52;
    float var_53;
    float var_54;
    df::float3 var_55;
    float var_56;
    df::float3 var_57;
    float var_58;
    df::float3 var_59;
    df::float3 var_60;
    df::float3 var_61;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_contact_body, var_0);
    var_2 = df::load(var_contact_point, var_0);
    var_3 = df::load(var_contact_dist, var_0);
    var_4 = df::load(var_contact_mat, var_0);
    var_6 = df::mul(var_4, var_5);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_materials, var_8);
    var_10 = df::mul(var_4, var_5);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_materials, var_12);
    var_14 = df::mul(var_4, var_5);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_materials, var_16);
    var_18 = df::mul(var_4, var_5);
    var_20 = df::add(var_18, var_19);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::load(var_rigid_x, var_1);
    var_23 = df::load(var_rigid_r, var_1);
    var_24 = df::load(var_rigid_v, var_1);
    var_25 = df::load(var_rigid_w, var_1);
    var_28 = df::float3(var_26, var_27, var_26);
    var_29 = df::rotate(var_23, var_2);
    var_30 = df::add(var_22, var_29);
    var_31 = df::mul(var_28, var_3);
    var_32 = df::sub(var_30, var_31);
    var_33 = df::sub(var_32, var_22);
    var_34 = df::cross(var_25, var_33);
    var_35 = df::add(var_24, var_34);
    var_36 = df::dot(var_28, var_32);
    var_37 = df::min(var_36, var_26);
    var_38 = df::dot(var_28, var_35);
    var_39 = df::mul(var_28, var_38);
    var_40 = df::sub(var_35, var_39);
    var_41 = df::mul(var_37, var_9);
    var_42 = df::min(var_38, var_26);
    var_43 = df::mul(var_42, var_13);
    var_44 = df::step(var_37);
    var_45 = df::mul(var_43, var_44);
    var_46 = df::add(var_41, var_45);
    var_47 = df::mul(var_21, var_46);
    var_48 = df::sub(var_26, var_47);
    var_49 = df::float3(var_17, var_26, var_26);
    var_50 = df::dot(var_49, var_40);
    var_51 = df::clamp(var_50, var_47, var_48);
    var_52 = df::float3(var_26, var_26, var_17);
    var_53 = df::dot(var_52, var_40);
    var_54 = df::clamp(var_53, var_47, var_48);
    var_55 = df::float3(var_51, var_26, var_54);
    var_56 = df::step(var_37);
    var_57 = df::mul(var_55, var_56);
    var_58 = df::add(var_41, var_45);
    var_59 = df::mul(var_28, var_58);
    var_60 = df::add(var_59, var_57);
    var_61 = df::cross(var_33, var_60);
    df::atomic_sub(var_rigid_f, var_1, var_60);
    df::atomic_sub(var_rigid_t, var_1, var_61);
    

}

void eval_rigid_contacts_cpu_kernel_backward(
	df::float3* var_rigid_x,
	quat* var_rigid_r,
	df::float3* var_rigid_v,
	df::float3* var_rigid_w,
	int* var_contact_body,
	df::float3* var_contact_point,
	float* var_contact_dist,
	int* var_contact_mat,
	float* var_materials,
	df::float3* var_rigid_f,
	df::float3* var_rigid_t, 
	df::float3* adj_rigid_x,
	quat* adj_rigid_r,
	df::float3* adj_rigid_v,
	df::float3* adj_rigid_w,
	int* adj_contact_body,
	df::float3* adj_contact_point,
	float* adj_contact_dist,
	int* adj_contact_mat,
	float* adj_materials,
	df::float3* adj_rigid_f,
	df::float3* adj_rigid_t)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    df::float3 var_2;
    float var_3;
    int var_4;
    const int var_5 = 4;
    int var_6;
    const int var_7 = 0;
    int var_8;
    float var_9;
    int var_10;
    const int var_11 = 1;
    int var_12;
    float var_13;
    int var_14;
    const int var_15 = 2;
    int var_16;
    float var_17;
    int var_18;
    const int var_19 = 3;
    int var_20;
    float var_21;
    df::float3 var_22;
    quat var_23;
    df::float3 var_24;
    df::float3 var_25;
    const float var_26 = 0.0;
    const float var_27 = 1.0;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    float var_36;
    float var_37;
    float var_38;
    df::float3 var_39;
    df::float3 var_40;
    float var_41;
    float var_42;
    float var_43;
    float var_44;
    float var_45;
    float var_46;
    float var_47;
    float var_48;
    df::float3 var_49;
    float var_50;
    float var_51;
    df::float3 var_52;
    float var_53;
    float var_54;
    df::float3 var_55;
    float var_56;
    df::float3 var_57;
    float var_58;
    df::float3 var_59;
    df::float3 var_60;
    df::float3 var_61;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    df::float3 adj_2 = 0;
    float adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    float adj_9 = 0;
    int adj_10 = 0;
    int adj_11 = 0;
    int adj_12 = 0;
    float adj_13 = 0;
    int adj_14 = 0;
    int adj_15 = 0;
    int adj_16 = 0;
    float adj_17 = 0;
    int adj_18 = 0;
    int adj_19 = 0;
    int adj_20 = 0;
    float adj_21 = 0;
    df::float3 adj_22 = 0;
    quat adj_23 = 0;
    df::float3 adj_24 = 0;
    df::float3 adj_25 = 0;
    float adj_26 = 0;
    float adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    df::float3 adj_32 = 0;
    df::float3 adj_33 = 0;
    df::float3 adj_34 = 0;
    df::float3 adj_35 = 0;
    float adj_36 = 0;
    float adj_37 = 0;
    float adj_38 = 0;
    df::float3 adj_39 = 0;
    df::float3 adj_40 = 0;
    float adj_41 = 0;
    float adj_42 = 0;
    float adj_43 = 0;
    float adj_44 = 0;
    float adj_45 = 0;
    float adj_46 = 0;
    float adj_47 = 0;
    float adj_48 = 0;
    df::float3 adj_49 = 0;
    float adj_50 = 0;
    float adj_51 = 0;
    df::float3 adj_52 = 0;
    float adj_53 = 0;
    float adj_54 = 0;
    df::float3 adj_55 = 0;
    float adj_56 = 0;
    df::float3 adj_57 = 0;
    float adj_58 = 0;
    df::float3 adj_59 = 0;
    df::float3 adj_60 = 0;
    df::float3 adj_61 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_contact_body, var_0);
    var_2 = df::load(var_contact_point, var_0);
    var_3 = df::load(var_contact_dist, var_0);
    var_4 = df::load(var_contact_mat, var_0);
    var_6 = df::mul(var_4, var_5);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_materials, var_8);
    var_10 = df::mul(var_4, var_5);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_materials, var_12);
    var_14 = df::mul(var_4, var_5);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_materials, var_16);
    var_18 = df::mul(var_4, var_5);
    var_20 = df::add(var_18, var_19);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::load(var_rigid_x, var_1);
    var_23 = df::load(var_rigid_r, var_1);
    var_24 = df::load(var_rigid_v, var_1);
    var_25 = df::load(var_rigid_w, var_1);
    var_28 = df::float3(var_26, var_27, var_26);
    var_29 = df::rotate(var_23, var_2);
    var_30 = df::add(var_22, var_29);
    var_31 = df::mul(var_28, var_3);
    var_32 = df::sub(var_30, var_31);
    var_33 = df::sub(var_32, var_22);
    var_34 = df::cross(var_25, var_33);
    var_35 = df::add(var_24, var_34);
    var_36 = df::dot(var_28, var_32);
    var_37 = df::min(var_36, var_26);
    var_38 = df::dot(var_28, var_35);
    var_39 = df::mul(var_28, var_38);
    var_40 = df::sub(var_35, var_39);
    var_41 = df::mul(var_37, var_9);
    var_42 = df::min(var_38, var_26);
    var_43 = df::mul(var_42, var_13);
    var_44 = df::step(var_37);
    var_45 = df::mul(var_43, var_44);
    var_46 = df::add(var_41, var_45);
    var_47 = df::mul(var_21, var_46);
    var_48 = df::sub(var_26, var_47);
    var_49 = df::float3(var_17, var_26, var_26);
    var_50 = df::dot(var_49, var_40);
    var_51 = df::clamp(var_50, var_47, var_48);
    var_52 = df::float3(var_26, var_26, var_17);
    var_53 = df::dot(var_52, var_40);
    var_54 = df::clamp(var_53, var_47, var_48);
    var_55 = df::float3(var_51, var_26, var_54);
    var_56 = df::step(var_37);
    var_57 = df::mul(var_55, var_56);
    var_58 = df::add(var_41, var_45);
    var_59 = df::mul(var_28, var_58);
    var_60 = df::add(var_59, var_57);
    var_61 = df::cross(var_33, var_60);
    df::atomic_sub(var_rigid_f, var_1, var_60);
    df::atomic_sub(var_rigid_t, var_1, var_61);
    //---------
    // reverse
    df::adj_atomic_sub(var_rigid_t, var_1, var_61, adj_rigid_t, adj_1, adj_61);
    df::adj_atomic_sub(var_rigid_f, var_1, var_60, adj_rigid_f, adj_1, adj_60);
    df::adj_cross(var_33, var_60, adj_33, adj_60, adj_61);
    df::adj_add(var_59, var_57, adj_59, adj_57, adj_60);
    df::adj_mul(var_28, var_58, adj_28, adj_58, adj_59);
    df::adj_add(var_41, var_45, adj_41, adj_45, adj_58);
    df::adj_mul(var_55, var_56, adj_55, adj_56, adj_57);
    df::adj_step(var_37, adj_37, adj_56);
    df::adj_float3(var_51, var_26, var_54, adj_51, adj_26, adj_54, adj_55);
    df::adj_clamp(var_53, var_47, var_48, adj_53, adj_47, adj_48, adj_54);
    df::adj_dot(var_52, var_40, adj_52, adj_40, adj_53);
    df::adj_float3(var_26, var_26, var_17, adj_26, adj_26, adj_17, adj_52);
    df::adj_clamp(var_50, var_47, var_48, adj_50, adj_47, adj_48, adj_51);
    df::adj_dot(var_49, var_40, adj_49, adj_40, adj_50);
    df::adj_float3(var_17, var_26, var_26, adj_17, adj_26, adj_26, adj_49);
    df::adj_sub(var_26, var_47, adj_26, adj_47, adj_48);
    df::adj_mul(var_21, var_46, adj_21, adj_46, adj_47);
    df::adj_add(var_41, var_45, adj_41, adj_45, adj_46);
    df::adj_mul(var_43, var_44, adj_43, adj_44, adj_45);
    df::adj_step(var_37, adj_37, adj_44);
    df::adj_mul(var_42, var_13, adj_42, adj_13, adj_43);
    df::adj_min(var_38, var_26, adj_38, adj_26, adj_42);
    df::adj_mul(var_37, var_9, adj_37, adj_9, adj_41);
    df::adj_sub(var_35, var_39, adj_35, adj_39, adj_40);
    df::adj_mul(var_28, var_38, adj_28, adj_38, adj_39);
    df::adj_dot(var_28, var_35, adj_28, adj_35, adj_38);
    df::adj_min(var_36, var_26, adj_36, adj_26, adj_37);
    df::adj_dot(var_28, var_32, adj_28, adj_32, adj_36);
    df::adj_add(var_24, var_34, adj_24, adj_34, adj_35);
    df::adj_cross(var_25, var_33, adj_25, adj_33, adj_34);
    df::adj_sub(var_32, var_22, adj_32, adj_22, adj_33);
    df::adj_sub(var_30, var_31, adj_30, adj_31, adj_32);
    df::adj_mul(var_28, var_3, adj_28, adj_3, adj_31);
    df::adj_add(var_22, var_29, adj_22, adj_29, adj_30);
    df::adj_rotate(var_23, var_2, adj_23, adj_2, adj_29);
    df::adj_float3(var_26, var_27, var_26, adj_26, adj_27, adj_26, adj_28);
    df::adj_load(var_rigid_w, var_1, adj_rigid_w, adj_1, adj_25);
    df::adj_load(var_rigid_v, var_1, adj_rigid_v, adj_1, adj_24);
    df::adj_load(var_rigid_r, var_1, adj_rigid_r, adj_1, adj_23);
    df::adj_load(var_rigid_x, var_1, adj_rigid_x, adj_1, adj_22);
    df::adj_load(var_materials, var_20, adj_materials, adj_20, adj_21);
    df::adj_add(var_18, var_19, adj_18, adj_19, adj_20);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_18);
    df::adj_load(var_materials, var_16, adj_materials, adj_16, adj_17);
    df::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_14);
    df::adj_load(var_materials, var_12, adj_materials, adj_12, adj_13);
    df::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_10);
    df::adj_load(var_materials, var_8, adj_materials, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_6);
    df::adj_load(var_contact_mat, var_0, adj_contact_mat, adj_0, adj_4);
    df::adj_load(var_contact_dist, var_0, adj_contact_dist, adj_0, adj_3);
    df::adj_load(var_contact_point, var_0, adj_contact_point, adj_0, adj_2);
    df::adj_load(var_contact_body, var_0, adj_contact_body, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_contacts_cpu_forward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_contacts_cpu_kernel_forward(
			cast<df::float3*>(var_rigid_x),
			cast<quat*>(var_rigid_r),
			cast<df::float3*>(var_rigid_v),
			cast<df::float3*>(var_rigid_w),
			cast<int*>(var_contact_body),
			cast<df::float3*>(var_contact_point),
			cast<float*>(var_contact_dist),
			cast<int*>(var_contact_mat),
			cast<float*>(var_materials),
			cast<df::float3*>(var_rigid_f),
			cast<df::float3*>(var_rigid_t));
    }
}

void eval_rigid_contacts_cpu_backward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t, 
	torch::Tensor adj_rigid_x,
	torch::Tensor adj_rigid_r,
	torch::Tensor adj_rigid_v,
	torch::Tensor adj_rigid_w,
	torch::Tensor adj_contact_body,
	torch::Tensor adj_contact_point,
	torch::Tensor adj_contact_dist,
	torch::Tensor adj_contact_mat,
	torch::Tensor adj_materials,
	torch::Tensor adj_rigid_f,
	torch::Tensor adj_rigid_t)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_contacts_cpu_kernel_backward(
			cast<df::float3*>(var_rigid_x),
			cast<quat*>(var_rigid_r),
			cast<df::float3*>(var_rigid_v),
			cast<df::float3*>(var_rigid_w),
			cast<int*>(var_contact_body),
			cast<df::float3*>(var_contact_point),
			cast<float*>(var_contact_dist),
			cast<int*>(var_contact_mat),
			cast<float*>(var_materials),
			cast<df::float3*>(var_rigid_f),
			cast<df::float3*>(var_rigid_t), 
			cast<df::float3*>(adj_rigid_x),
			cast<quat*>(adj_rigid_r),
			cast<df::float3*>(adj_rigid_v),
			cast<df::float3*>(adj_rigid_w),
			cast<int*>(adj_contact_body),
			cast<df::float3*>(adj_contact_point),
			cast<float*>(adj_contact_dist),
			cast<int*>(adj_contact_mat),
			cast<float*>(adj_materials),
			cast<df::float3*>(adj_rigid_f),
			cast<df::float3*>(adj_rigid_t));
    }
}



// Python entry points
void eval_rigid_contacts_cpu_forward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t);

void eval_rigid_contacts_cpu_backward(int dim, 
	torch::Tensor var_rigid_x,
	torch::Tensor var_rigid_r,
	torch::Tensor var_rigid_v,
	torch::Tensor var_rigid_w,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_rigid_f,
	torch::Tensor var_rigid_t, 
	torch::Tensor adj_rigid_x,
	torch::Tensor adj_rigid_r,
	torch::Tensor adj_rigid_v,
	torch::Tensor adj_rigid_w,
	torch::Tensor adj_contact_body,
	torch::Tensor adj_contact_point,
	torch::Tensor adj_contact_dist,
	torch::Tensor adj_contact_mat,
	torch::Tensor adj_materials,
	torch::Tensor adj_rigid_f,
	torch::Tensor adj_rigid_t);


void eval_rigid_contacts_art_cpu_kernel_forward(
	spatial_transform* var_body_X_s,
	spatial_vector* var_body_v_s,
	int* var_contact_body,
	df::float3* var_contact_point,
	float* var_contact_dist,
	int* var_contact_mat,
	float* var_materials,
	spatial_vector* var_body_f_s)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    df::float3 var_2;
    float var_3;
    int var_4;
    const int var_5 = 4;
    int var_6;
    const int var_7 = 0;
    int var_8;
    float var_9;
    int var_10;
    const int var_11 = 1;
    int var_12;
    float var_13;
    int var_14;
    const int var_15 = 2;
    int var_16;
    float var_17;
    int var_18;
    const int var_19 = 3;
    int var_20;
    float var_21;
    spatial_transform var_22;
    spatial_vector var_23;
    const float var_24 = 0.0;
    const float var_25 = 1.0;
    df::float3 var_26;
    df::float3 var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    float var_34;
    bool var_35;
    float var_36;
    df::float3 var_37;
    df::float3 var_38;
    float var_39;
    float var_40;
    float var_41;
    float var_42;
    float var_43;
    float var_44;
    float var_45;
    float var_46;
    float var_47;
    float var_48;
    df::float3 var_49;
    float var_50;
    float var_51;
    df::float3 var_52;
    float var_53;
    float var_54;
    df::float3 var_55;
    float var_56;
    float var_57;
    float var_58;
    float var_59;
    float var_60;
    float var_61;
    df::float3 var_62;
    float var_63;
    df::float3 var_64;
    float var_65;
    df::float3 var_66;
    df::float3 var_67;
    df::float3 var_68;
    spatial_vector var_69;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_contact_body, var_0);
    var_2 = df::load(var_contact_point, var_0);
    var_3 = df::load(var_contact_dist, var_0);
    var_4 = df::load(var_contact_mat, var_0);
    var_6 = df::mul(var_4, var_5);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_materials, var_8);
    var_10 = df::mul(var_4, var_5);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_materials, var_12);
    var_14 = df::mul(var_4, var_5);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_materials, var_16);
    var_18 = df::mul(var_4, var_5);
    var_20 = df::add(var_18, var_19);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::load(var_body_X_s, var_1);
    var_23 = df::load(var_body_v_s, var_1);
    var_26 = df::float3(var_24, var_25, var_24);
    var_27 = df::spatial_transform_point(var_22, var_2);
    var_28 = df::mul(var_26, var_3);
    var_29 = df::sub(var_27, var_28);
    var_30 = df::spatial_top(var_23);
    var_31 = df::spatial_bottom(var_23);
    var_32 = df::cross(var_30, var_29);
    var_33 = df::add(var_31, var_32);
    var_34 = df::dot(var_26, var_29);
    var_35 = (var_34 >= var_24);
    if (var_35) {
    	return;
    }
    var_36 = df::dot(var_26, var_33);
    var_37 = df::mul(var_26, var_36);
    var_38 = df::sub(var_33, var_37);
    var_39 = df::mul(var_34, var_9);
    var_40 = df::min(var_36, var_24);
    var_41 = df::mul(var_40, var_13);
    var_42 = df::step(var_34);
    var_43 = df::mul(var_41, var_42);
    var_44 = df::sub(var_24, var_34);
    var_45 = df::mul(var_43, var_44);
    var_46 = df::add(var_39, var_45);
    var_47 = df::mul(var_21, var_46);
    var_48 = df::sub(var_24, var_47);
    var_49 = df::float3(var_17, var_24, var_24);
    var_50 = df::dot(var_49, var_38);
    var_51 = df::clamp(var_50, var_47, var_48);
    var_52 = df::float3(var_24, var_24, var_17);
    var_53 = df::dot(var_52, var_38);
    var_54 = df::clamp(var_53, var_47, var_48);
    var_55 = df::normalize(var_38);
    var_56 = df::length(var_38);
    var_57 = df::mul(var_17, var_56);
    var_58 = df::mul(var_21, var_34);
    var_59 = df::mul(var_58, var_9);
    var_60 = df::sub(var_24, var_59);
    var_61 = df::min(var_57, var_60);
    var_62 = df::mul(var_55, var_61);
    var_63 = df::step(var_34);
    var_64 = df::mul(var_62, var_63);
    var_65 = df::add(var_39, var_45);
    var_66 = df::mul(var_26, var_65);
    var_67 = df::add(var_66, var_64);
    var_68 = df::cross(var_29, var_67);
    var_69 = df::spatial_vector(var_68, var_67);
    df::atomic_add(var_body_f_s, var_1, var_69);
    

}

void eval_rigid_contacts_art_cpu_kernel_backward(
	spatial_transform* var_body_X_s,
	spatial_vector* var_body_v_s,
	int* var_contact_body,
	df::float3* var_contact_point,
	float* var_contact_dist,
	int* var_contact_mat,
	float* var_materials,
	spatial_vector* var_body_f_s, 
	spatial_transform* adj_body_X_s,
	spatial_vector* adj_body_v_s,
	int* adj_contact_body,
	df::float3* adj_contact_point,
	float* adj_contact_dist,
	int* adj_contact_mat,
	float* adj_materials,
	spatial_vector* adj_body_f_s)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    df::float3 var_2;
    float var_3;
    int var_4;
    const int var_5 = 4;
    int var_6;
    const int var_7 = 0;
    int var_8;
    float var_9;
    int var_10;
    const int var_11 = 1;
    int var_12;
    float var_13;
    int var_14;
    const int var_15 = 2;
    int var_16;
    float var_17;
    int var_18;
    const int var_19 = 3;
    int var_20;
    float var_21;
    spatial_transform var_22;
    spatial_vector var_23;
    const float var_24 = 0.0;
    const float var_25 = 1.0;
    df::float3 var_26;
    df::float3 var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    float var_34;
    bool var_35;
    float var_36;
    df::float3 var_37;
    df::float3 var_38;
    float var_39;
    float var_40;
    float var_41;
    float var_42;
    float var_43;
    float var_44;
    float var_45;
    float var_46;
    float var_47;
    float var_48;
    df::float3 var_49;
    float var_50;
    float var_51;
    df::float3 var_52;
    float var_53;
    float var_54;
    df::float3 var_55;
    float var_56;
    float var_57;
    float var_58;
    float var_59;
    float var_60;
    float var_61;
    df::float3 var_62;
    float var_63;
    df::float3 var_64;
    float var_65;
    df::float3 var_66;
    df::float3 var_67;
    df::float3 var_68;
    spatial_vector var_69;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    df::float3 adj_2 = 0;
    float adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    float adj_9 = 0;
    int adj_10 = 0;
    int adj_11 = 0;
    int adj_12 = 0;
    float adj_13 = 0;
    int adj_14 = 0;
    int adj_15 = 0;
    int adj_16 = 0;
    float adj_17 = 0;
    int adj_18 = 0;
    int adj_19 = 0;
    int adj_20 = 0;
    float adj_21 = 0;
    spatial_transform adj_22 = 0;
    spatial_vector adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    df::float3 adj_26 = 0;
    df::float3 adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    df::float3 adj_32 = 0;
    df::float3 adj_33 = 0;
    float adj_34 = 0;
    bool adj_35 = 0;
    float adj_36 = 0;
    df::float3 adj_37 = 0;
    df::float3 adj_38 = 0;
    float adj_39 = 0;
    float adj_40 = 0;
    float adj_41 = 0;
    float adj_42 = 0;
    float adj_43 = 0;
    float adj_44 = 0;
    float adj_45 = 0;
    float adj_46 = 0;
    float adj_47 = 0;
    float adj_48 = 0;
    df::float3 adj_49 = 0;
    float adj_50 = 0;
    float adj_51 = 0;
    df::float3 adj_52 = 0;
    float adj_53 = 0;
    float adj_54 = 0;
    df::float3 adj_55 = 0;
    float adj_56 = 0;
    float adj_57 = 0;
    float adj_58 = 0;
    float adj_59 = 0;
    float adj_60 = 0;
    float adj_61 = 0;
    df::float3 adj_62 = 0;
    float adj_63 = 0;
    df::float3 adj_64 = 0;
    float adj_65 = 0;
    df::float3 adj_66 = 0;
    df::float3 adj_67 = 0;
    df::float3 adj_68 = 0;
    spatial_vector adj_69 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_contact_body, var_0);
    var_2 = df::load(var_contact_point, var_0);
    var_3 = df::load(var_contact_dist, var_0);
    var_4 = df::load(var_contact_mat, var_0);
    var_6 = df::mul(var_4, var_5);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_materials, var_8);
    var_10 = df::mul(var_4, var_5);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_materials, var_12);
    var_14 = df::mul(var_4, var_5);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_materials, var_16);
    var_18 = df::mul(var_4, var_5);
    var_20 = df::add(var_18, var_19);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::load(var_body_X_s, var_1);
    var_23 = df::load(var_body_v_s, var_1);
    var_26 = df::float3(var_24, var_25, var_24);
    var_27 = df::spatial_transform_point(var_22, var_2);
    var_28 = df::mul(var_26, var_3);
    var_29 = df::sub(var_27, var_28);
    var_30 = df::spatial_top(var_23);
    var_31 = df::spatial_bottom(var_23);
    var_32 = df::cross(var_30, var_29);
    var_33 = df::add(var_31, var_32);
    var_34 = df::dot(var_26, var_29);
    var_35 = (var_34 >= var_24);
    if (var_35) {
    	goto label0;
    }
    var_36 = df::dot(var_26, var_33);
    var_37 = df::mul(var_26, var_36);
    var_38 = df::sub(var_33, var_37);
    var_39 = df::mul(var_34, var_9);
    var_40 = df::min(var_36, var_24);
    var_41 = df::mul(var_40, var_13);
    var_42 = df::step(var_34);
    var_43 = df::mul(var_41, var_42);
    var_44 = df::sub(var_24, var_34);
    var_45 = df::mul(var_43, var_44);
    var_46 = df::add(var_39, var_45);
    var_47 = df::mul(var_21, var_46);
    var_48 = df::sub(var_24, var_47);
    var_49 = df::float3(var_17, var_24, var_24);
    var_50 = df::dot(var_49, var_38);
    var_51 = df::clamp(var_50, var_47, var_48);
    var_52 = df::float3(var_24, var_24, var_17);
    var_53 = df::dot(var_52, var_38);
    var_54 = df::clamp(var_53, var_47, var_48);
    var_55 = df::normalize(var_38);
    var_56 = df::length(var_38);
    var_57 = df::mul(var_17, var_56);
    var_58 = df::mul(var_21, var_34);
    var_59 = df::mul(var_58, var_9);
    var_60 = df::sub(var_24, var_59);
    var_61 = df::min(var_57, var_60);
    var_62 = df::mul(var_55, var_61);
    var_63 = df::step(var_34);
    var_64 = df::mul(var_62, var_63);
    var_65 = df::add(var_39, var_45);
    var_66 = df::mul(var_26, var_65);
    var_67 = df::add(var_66, var_64);
    var_68 = df::cross(var_29, var_67);
    var_69 = df::spatial_vector(var_68, var_67);
    df::atomic_add(var_body_f_s, var_1, var_69);
    //---------
    // reverse
    df::adj_atomic_add(var_body_f_s, var_1, var_69, adj_body_f_s, adj_1, adj_69);
    df::adj_spatial_vector(var_68, var_67, adj_68, adj_67, adj_69);
    df::adj_cross(var_29, var_67, adj_29, adj_67, adj_68);
    df::adj_add(var_66, var_64, adj_66, adj_64, adj_67);
    df::adj_mul(var_26, var_65, adj_26, adj_65, adj_66);
    df::adj_add(var_39, var_45, adj_39, adj_45, adj_65);
    df::adj_mul(var_62, var_63, adj_62, adj_63, adj_64);
    df::adj_step(var_34, adj_34, adj_63);
    df::adj_mul(var_55, var_61, adj_55, adj_61, adj_62);
    df::adj_min(var_57, var_60, adj_57, adj_60, adj_61);
    df::adj_sub(var_24, var_59, adj_24, adj_59, adj_60);
    df::adj_mul(var_58, var_9, adj_58, adj_9, adj_59);
    df::adj_mul(var_21, var_34, adj_21, adj_34, adj_58);
    df::adj_mul(var_17, var_56, adj_17, adj_56, adj_57);
    df::adj_length(var_38, adj_38, adj_56);
    df::adj_normalize(var_38, adj_38, adj_55);
    df::adj_clamp(var_53, var_47, var_48, adj_53, adj_47, adj_48, adj_54);
    df::adj_dot(var_52, var_38, adj_52, adj_38, adj_53);
    df::adj_float3(var_24, var_24, var_17, adj_24, adj_24, adj_17, adj_52);
    df::adj_clamp(var_50, var_47, var_48, adj_50, adj_47, adj_48, adj_51);
    df::adj_dot(var_49, var_38, adj_49, adj_38, adj_50);
    df::adj_float3(var_17, var_24, var_24, adj_17, adj_24, adj_24, adj_49);
    df::adj_sub(var_24, var_47, adj_24, adj_47, adj_48);
    df::adj_mul(var_21, var_46, adj_21, adj_46, adj_47);
    df::adj_add(var_39, var_45, adj_39, adj_45, adj_46);
    df::adj_mul(var_43, var_44, adj_43, adj_44, adj_45);
    df::adj_sub(var_24, var_34, adj_24, adj_34, adj_44);
    df::adj_mul(var_41, var_42, adj_41, adj_42, adj_43);
    df::adj_step(var_34, adj_34, adj_42);
    df::adj_mul(var_40, var_13, adj_40, adj_13, adj_41);
    df::adj_min(var_36, var_24, adj_36, adj_24, adj_40);
    df::adj_mul(var_34, var_9, adj_34, adj_9, adj_39);
    df::adj_sub(var_33, var_37, adj_33, adj_37, adj_38);
    df::adj_mul(var_26, var_36, adj_26, adj_36, adj_37);
    df::adj_dot(var_26, var_33, adj_26, adj_33, adj_36);
    if (var_35) {
    	label0:;
    }
    df::adj_dot(var_26, var_29, adj_26, adj_29, adj_34);
    df::adj_add(var_31, var_32, adj_31, adj_32, adj_33);
    df::adj_cross(var_30, var_29, adj_30, adj_29, adj_32);
    df::adj_spatial_bottom(var_23, adj_23, adj_31);
    df::adj_spatial_top(var_23, adj_23, adj_30);
    df::adj_sub(var_27, var_28, adj_27, adj_28, adj_29);
    df::adj_mul(var_26, var_3, adj_26, adj_3, adj_28);
    df::adj_spatial_transform_point(var_22, var_2, adj_22, adj_2, adj_27);
    df::adj_float3(var_24, var_25, var_24, adj_24, adj_25, adj_24, adj_26);
    df::adj_load(var_body_v_s, var_1, adj_body_v_s, adj_1, adj_23);
    df::adj_load(var_body_X_s, var_1, adj_body_X_s, adj_1, adj_22);
    df::adj_load(var_materials, var_20, adj_materials, adj_20, adj_21);
    df::adj_add(var_18, var_19, adj_18, adj_19, adj_20);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_18);
    df::adj_load(var_materials, var_16, adj_materials, adj_16, adj_17);
    df::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_14);
    df::adj_load(var_materials, var_12, adj_materials, adj_12, adj_13);
    df::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_10);
    df::adj_load(var_materials, var_8, adj_materials, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_4, var_5, adj_4, adj_5, adj_6);
    df::adj_load(var_contact_mat, var_0, adj_contact_mat, adj_0, adj_4);
    df::adj_load(var_contact_dist, var_0, adj_contact_dist, adj_0, adj_3);
    df::adj_load(var_contact_point, var_0, adj_contact_point, adj_0, adj_2);
    df::adj_load(var_contact_body, var_0, adj_contact_body, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_contacts_art_cpu_forward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_body_f_s)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_contacts_art_cpu_kernel_forward(
			cast<spatial_transform*>(var_body_X_s),
			cast<spatial_vector*>(var_body_v_s),
			cast<int*>(var_contact_body),
			cast<df::float3*>(var_contact_point),
			cast<float*>(var_contact_dist),
			cast<int*>(var_contact_mat),
			cast<float*>(var_materials),
			cast<spatial_vector*>(var_body_f_s));
    }
}

void eval_rigid_contacts_art_cpu_backward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_body_f_s, 
	torch::Tensor adj_body_X_s,
	torch::Tensor adj_body_v_s,
	torch::Tensor adj_contact_body,
	torch::Tensor adj_contact_point,
	torch::Tensor adj_contact_dist,
	torch::Tensor adj_contact_mat,
	torch::Tensor adj_materials,
	torch::Tensor adj_body_f_s)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_contacts_art_cpu_kernel_backward(
			cast<spatial_transform*>(var_body_X_s),
			cast<spatial_vector*>(var_body_v_s),
			cast<int*>(var_contact_body),
			cast<df::float3*>(var_contact_point),
			cast<float*>(var_contact_dist),
			cast<int*>(var_contact_mat),
			cast<float*>(var_materials),
			cast<spatial_vector*>(var_body_f_s), 
			cast<spatial_transform*>(adj_body_X_s),
			cast<spatial_vector*>(adj_body_v_s),
			cast<int*>(adj_contact_body),
			cast<df::float3*>(adj_contact_point),
			cast<float*>(adj_contact_dist),
			cast<int*>(adj_contact_mat),
			cast<float*>(adj_materials),
			cast<spatial_vector*>(adj_body_f_s));
    }
}



// Python entry points
void eval_rigid_contacts_art_cpu_forward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_body_f_s);

void eval_rigid_contacts_art_cpu_backward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_contact_body,
	torch::Tensor var_contact_point,
	torch::Tensor var_contact_dist,
	torch::Tensor var_contact_mat,
	torch::Tensor var_materials,
	torch::Tensor var_body_f_s, 
	torch::Tensor adj_body_X_s,
	torch::Tensor adj_body_v_s,
	torch::Tensor adj_contact_body,
	torch::Tensor adj_contact_point,
	torch::Tensor adj_contact_dist,
	torch::Tensor adj_contact_mat,
	torch::Tensor adj_materials,
	torch::Tensor adj_body_f_s);


void eval_muscles_cpu_kernel_forward(
	spatial_transform* var_body_X_s,
	spatial_vector* var_body_v_s,
	int* var_muscle_start,
	float* var_muscle_params,
	int* var_muscle_links,
	df::float3* var_muscle_points,
	float* var_muscle_activation,
	spatial_vector* var_body_f_s)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    float var_6;
    int var_7;
    int var_8;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_muscle_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_muscle_start, var_3);
    var_5 = df::sub(var_4, var_2);
    var_6 = df::load(var_muscle_activation, var_0);
    for (var_7=var_1; var_7 < var_5; ++var_7) {
    	var_8 = compute_muscle_force_cpu_func(var_7, var_body_X_s, var_body_v_s, var_muscle_links, var_muscle_points, var_6, var_body_f_s);
    }
    

}

void eval_muscles_cpu_kernel_backward(
	spatial_transform* var_body_X_s,
	spatial_vector* var_body_v_s,
	int* var_muscle_start,
	float* var_muscle_params,
	int* var_muscle_links,
	df::float3* var_muscle_points,
	float* var_muscle_activation,
	spatial_vector* var_body_f_s, 
	spatial_transform* adj_body_X_s,
	spatial_vector* adj_body_v_s,
	int* adj_muscle_start,
	float* adj_muscle_params,
	int* adj_muscle_links,
	df::float3* adj_muscle_points,
	float* adj_muscle_activation,
	spatial_vector* adj_body_f_s)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    float var_6;
    int var_7;
    int var_8;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    float adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_muscle_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_muscle_start, var_3);
    var_5 = df::sub(var_4, var_2);
    var_6 = df::load(var_muscle_activation, var_0);
    if (false) {
    	var_8 = compute_muscle_force_cpu_func(var_7, var_body_X_s, var_body_v_s, var_muscle_links, var_muscle_points, var_6, var_body_f_s);
    }
    //---------
    // reverse
    for (var_7=var_5-1; var_7 >= var_1; --var_7) {
    	adj_compute_muscle_force_cpu_func(var_7, var_body_X_s, var_body_v_s, var_muscle_links, var_muscle_points, var_6, var_body_f_s, adj_7, adj_body_X_s, adj_body_v_s, adj_muscle_links, adj_muscle_points, adj_6, adj_body_f_s, adj_8);
    }
    df::adj_load(var_muscle_activation, var_0, adj_muscle_activation, adj_0, adj_6);
    df::adj_sub(var_4, var_2, adj_4, adj_2, adj_5);
    df::adj_load(var_muscle_start, var_3, adj_muscle_start, adj_3, adj_4);
    df::adj_add(var_0, var_2, adj_0, adj_2, adj_3);
    df::adj_load(var_muscle_start, var_0, adj_muscle_start, adj_0, adj_1);
    return;

}



// Python entry points
void eval_muscles_cpu_forward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_muscle_start,
	torch::Tensor var_muscle_params,
	torch::Tensor var_muscle_links,
	torch::Tensor var_muscle_points,
	torch::Tensor var_muscle_activation,
	torch::Tensor var_body_f_s)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_muscles_cpu_kernel_forward(
			cast<spatial_transform*>(var_body_X_s),
			cast<spatial_vector*>(var_body_v_s),
			cast<int*>(var_muscle_start),
			cast<float*>(var_muscle_params),
			cast<int*>(var_muscle_links),
			cast<df::float3*>(var_muscle_points),
			cast<float*>(var_muscle_activation),
			cast<spatial_vector*>(var_body_f_s));
    }
}

void eval_muscles_cpu_backward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_muscle_start,
	torch::Tensor var_muscle_params,
	torch::Tensor var_muscle_links,
	torch::Tensor var_muscle_points,
	torch::Tensor var_muscle_activation,
	torch::Tensor var_body_f_s, 
	torch::Tensor adj_body_X_s,
	torch::Tensor adj_body_v_s,
	torch::Tensor adj_muscle_start,
	torch::Tensor adj_muscle_params,
	torch::Tensor adj_muscle_links,
	torch::Tensor adj_muscle_points,
	torch::Tensor adj_muscle_activation,
	torch::Tensor adj_body_f_s)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_muscles_cpu_kernel_backward(
			cast<spatial_transform*>(var_body_X_s),
			cast<spatial_vector*>(var_body_v_s),
			cast<int*>(var_muscle_start),
			cast<float*>(var_muscle_params),
			cast<int*>(var_muscle_links),
			cast<df::float3*>(var_muscle_points),
			cast<float*>(var_muscle_activation),
			cast<spatial_vector*>(var_body_f_s), 
			cast<spatial_transform*>(adj_body_X_s),
			cast<spatial_vector*>(adj_body_v_s),
			cast<int*>(adj_muscle_start),
			cast<float*>(adj_muscle_params),
			cast<int*>(adj_muscle_links),
			cast<df::float3*>(adj_muscle_points),
			cast<float*>(adj_muscle_activation),
			cast<spatial_vector*>(adj_body_f_s));
    }
}



// Python entry points
void eval_muscles_cpu_forward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_muscle_start,
	torch::Tensor var_muscle_params,
	torch::Tensor var_muscle_links,
	torch::Tensor var_muscle_points,
	torch::Tensor var_muscle_activation,
	torch::Tensor var_body_f_s);

void eval_muscles_cpu_backward(int dim, 
	torch::Tensor var_body_X_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_muscle_start,
	torch::Tensor var_muscle_params,
	torch::Tensor var_muscle_links,
	torch::Tensor var_muscle_points,
	torch::Tensor var_muscle_activation,
	torch::Tensor var_body_f_s, 
	torch::Tensor adj_body_X_s,
	torch::Tensor adj_body_v_s,
	torch::Tensor adj_muscle_start,
	torch::Tensor adj_muscle_params,
	torch::Tensor adj_muscle_links,
	torch::Tensor adj_muscle_points,
	torch::Tensor adj_muscle_activation,
	torch::Tensor adj_body_f_s);


void eval_rigid_fk_cpu_kernel_forward(
	int* var_articulation_start,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	spatial_transform* var_joint_X_pj,
	spatial_transform* var_joint_X_cm,
	df::float3* var_joint_axis,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    for (var_5=var_1; var_5 < var_4; ++var_5) {
    	var_6 = compute_link_transform_cpu_func(var_5, var_joint_type, var_joint_parent, var_joint_q_start, var_joint_qd_start, var_joint_q, var_joint_X_pj, var_joint_X_cm, var_joint_axis, var_body_X_sc, var_body_X_sm);
    }
    

}

void eval_rigid_fk_cpu_kernel_backward(
	int* var_articulation_start,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	spatial_transform* var_joint_X_pj,
	spatial_transform* var_joint_X_cm,
	df::float3* var_joint_axis,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm, 
	int* adj_articulation_start,
	int* adj_joint_type,
	int* adj_joint_parent,
	int* adj_joint_q_start,
	int* adj_joint_qd_start,
	float* adj_joint_q,
	spatial_transform* adj_joint_X_pj,
	spatial_transform* adj_joint_X_cm,
	df::float3* adj_joint_axis,
	spatial_transform* adj_body_X_sc,
	spatial_transform* adj_body_X_sm)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    if (false) {
    	var_6 = compute_link_transform_cpu_func(var_5, var_joint_type, var_joint_parent, var_joint_q_start, var_joint_qd_start, var_joint_q, var_joint_X_pj, var_joint_X_cm, var_joint_axis, var_body_X_sc, var_body_X_sm);
    }
    //---------
    // reverse
    for (var_5=var_4-1; var_5 >= var_1; --var_5) {
    	adj_compute_link_transform_cpu_func(var_5, var_joint_type, var_joint_parent, var_joint_q_start, var_joint_qd_start, var_joint_q, var_joint_X_pj, var_joint_X_cm, var_joint_axis, var_body_X_sc, var_body_X_sm, adj_5, adj_joint_type, adj_joint_parent, adj_joint_q_start, adj_joint_qd_start, adj_joint_q, adj_joint_X_pj, adj_joint_X_cm, adj_joint_axis, adj_body_X_sc, adj_body_X_sm, adj_6);
    }
    df::adj_load(var_articulation_start, var_3, adj_articulation_start, adj_3, adj_4);
    df::adj_add(var_0, var_2, adj_0, adj_2, adj_3);
    df::adj_load(var_articulation_start, var_0, adj_articulation_start, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_fk_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_joint_X_cm,
	torch::Tensor var_joint_axis,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_fk_cpu_kernel_forward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<spatial_transform*>(var_joint_X_pj),
			cast<spatial_transform*>(var_joint_X_cm),
			cast<df::float3*>(var_joint_axis),
			cast<spatial_transform*>(var_body_X_sc),
			cast<spatial_transform*>(var_body_X_sm));
    }
}

void eval_rigid_fk_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_joint_X_cm,
	torch::Tensor var_joint_axis,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_X_pj,
	torch::Tensor adj_joint_X_cm,
	torch::Tensor adj_joint_axis,
	torch::Tensor adj_body_X_sc,
	torch::Tensor adj_body_X_sm)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_fk_cpu_kernel_backward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<spatial_transform*>(var_joint_X_pj),
			cast<spatial_transform*>(var_joint_X_cm),
			cast<df::float3*>(var_joint_axis),
			cast<spatial_transform*>(var_body_X_sc),
			cast<spatial_transform*>(var_body_X_sm), 
			cast<int*>(adj_articulation_start),
			cast<int*>(adj_joint_type),
			cast<int*>(adj_joint_parent),
			cast<int*>(adj_joint_q_start),
			cast<int*>(adj_joint_qd_start),
			cast<float*>(adj_joint_q),
			cast<spatial_transform*>(adj_joint_X_pj),
			cast<spatial_transform*>(adj_joint_X_cm),
			cast<df::float3*>(adj_joint_axis),
			cast<spatial_transform*>(adj_body_X_sc),
			cast<spatial_transform*>(adj_body_X_sm));
    }
}



// Python entry points
void eval_rigid_fk_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_joint_X_cm,
	torch::Tensor var_joint_axis,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm);

void eval_rigid_fk_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_joint_X_cm,
	torch::Tensor var_joint_axis,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_X_pj,
	torch::Tensor adj_joint_X_cm,
	torch::Tensor adj_joint_axis,
	torch::Tensor adj_body_X_sc,
	torch::Tensor adj_body_X_sm);


void eval_rigid_id_cpu_kernel_forward(
	int* var_articulation_start,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	df::float3* var_joint_axis,
	float* var_joint_target_ke,
	float* var_joint_target_kd,
	spatial_matrix* var_body_I_m,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm,
	spatial_transform* var_joint_X_pj,
	df::float3* var_gravity,
	spatial_vector* var_joint_S_s,
	spatial_matrix* var_body_I_s,
	spatial_vector* var_body_v_s,
	spatial_vector* var_body_f_s,
	spatial_vector* var_body_a_s)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    int var_7;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    for (var_6=var_1; var_6 < var_4; ++var_6) {
    	var_7 = compute_link_velocity_cpu_func(var_6, var_joint_type, var_joint_parent, var_joint_qd_start, var_joint_qd, var_joint_axis, var_body_I_m, var_body_X_sc, var_body_X_sm, var_joint_X_pj, var_gravity, var_joint_S_s, var_body_I_s, var_body_v_s, var_body_f_s, var_body_a_s);
    }
    

}

void eval_rigid_id_cpu_kernel_backward(
	int* var_articulation_start,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	df::float3* var_joint_axis,
	float* var_joint_target_ke,
	float* var_joint_target_kd,
	spatial_matrix* var_body_I_m,
	spatial_transform* var_body_X_sc,
	spatial_transform* var_body_X_sm,
	spatial_transform* var_joint_X_pj,
	df::float3* var_gravity,
	spatial_vector* var_joint_S_s,
	spatial_matrix* var_body_I_s,
	spatial_vector* var_body_v_s,
	spatial_vector* var_body_f_s,
	spatial_vector* var_body_a_s, 
	int* adj_articulation_start,
	int* adj_joint_type,
	int* adj_joint_parent,
	int* adj_joint_q_start,
	int* adj_joint_qd_start,
	float* adj_joint_q,
	float* adj_joint_qd,
	df::float3* adj_joint_axis,
	float* adj_joint_target_ke,
	float* adj_joint_target_kd,
	spatial_matrix* adj_body_I_m,
	spatial_transform* adj_body_X_sc,
	spatial_transform* adj_body_X_sm,
	spatial_transform* adj_joint_X_pj,
	df::float3* adj_gravity,
	spatial_vector* adj_joint_S_s,
	spatial_matrix* adj_body_I_s,
	spatial_vector* adj_body_v_s,
	spatial_vector* adj_body_f_s,
	spatial_vector* adj_body_a_s)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    int var_7;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    if (false) {
    	var_7 = compute_link_velocity_cpu_func(var_6, var_joint_type, var_joint_parent, var_joint_qd_start, var_joint_qd, var_joint_axis, var_body_I_m, var_body_X_sc, var_body_X_sm, var_joint_X_pj, var_gravity, var_joint_S_s, var_body_I_s, var_body_v_s, var_body_f_s, var_body_a_s);
    }
    //---------
    // reverse
    for (var_6=var_4-1; var_6 >= var_1; --var_6) {
    	adj_compute_link_velocity_cpu_func(var_6, var_joint_type, var_joint_parent, var_joint_qd_start, var_joint_qd, var_joint_axis, var_body_I_m, var_body_X_sc, var_body_X_sm, var_joint_X_pj, var_gravity, var_joint_S_s, var_body_I_s, var_body_v_s, var_body_f_s, var_body_a_s, adj_6, adj_joint_type, adj_joint_parent, adj_joint_qd_start, adj_joint_qd, adj_joint_axis, adj_body_I_m, adj_body_X_sc, adj_body_X_sm, adj_joint_X_pj, adj_gravity, adj_joint_S_s, adj_body_I_s, adj_body_v_s, adj_body_f_s, adj_body_a_s, adj_7);
    }
    df::adj_sub(var_4, var_1, adj_4, adj_1, adj_5);
    df::adj_load(var_articulation_start, var_3, adj_articulation_start, adj_3, adj_4);
    df::adj_add(var_0, var_2, adj_0, adj_2, adj_3);
    df::adj_load(var_articulation_start, var_0, adj_articulation_start, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_id_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_body_I_m,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_gravity,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_I_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_body_f_s,
	torch::Tensor var_body_a_s)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_id_cpu_kernel_forward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<float*>(var_joint_qd),
			cast<df::float3*>(var_joint_axis),
			cast<float*>(var_joint_target_ke),
			cast<float*>(var_joint_target_kd),
			cast<spatial_matrix*>(var_body_I_m),
			cast<spatial_transform*>(var_body_X_sc),
			cast<spatial_transform*>(var_body_X_sm),
			cast<spatial_transform*>(var_joint_X_pj),
			cast<df::float3*>(var_gravity),
			cast<spatial_vector*>(var_joint_S_s),
			cast<spatial_matrix*>(var_body_I_s),
			cast<spatial_vector*>(var_body_v_s),
			cast<spatial_vector*>(var_body_f_s),
			cast<spatial_vector*>(var_body_a_s));
    }
}

void eval_rigid_id_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_body_I_m,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_gravity,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_I_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_body_f_s,
	torch::Tensor var_body_a_s, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_qd,
	torch::Tensor adj_joint_axis,
	torch::Tensor adj_joint_target_ke,
	torch::Tensor adj_joint_target_kd,
	torch::Tensor adj_body_I_m,
	torch::Tensor adj_body_X_sc,
	torch::Tensor adj_body_X_sm,
	torch::Tensor adj_joint_X_pj,
	torch::Tensor adj_gravity,
	torch::Tensor adj_joint_S_s,
	torch::Tensor adj_body_I_s,
	torch::Tensor adj_body_v_s,
	torch::Tensor adj_body_f_s,
	torch::Tensor adj_body_a_s)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_id_cpu_kernel_backward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<float*>(var_joint_qd),
			cast<df::float3*>(var_joint_axis),
			cast<float*>(var_joint_target_ke),
			cast<float*>(var_joint_target_kd),
			cast<spatial_matrix*>(var_body_I_m),
			cast<spatial_transform*>(var_body_X_sc),
			cast<spatial_transform*>(var_body_X_sm),
			cast<spatial_transform*>(var_joint_X_pj),
			cast<df::float3*>(var_gravity),
			cast<spatial_vector*>(var_joint_S_s),
			cast<spatial_matrix*>(var_body_I_s),
			cast<spatial_vector*>(var_body_v_s),
			cast<spatial_vector*>(var_body_f_s),
			cast<spatial_vector*>(var_body_a_s), 
			cast<int*>(adj_articulation_start),
			cast<int*>(adj_joint_type),
			cast<int*>(adj_joint_parent),
			cast<int*>(adj_joint_q_start),
			cast<int*>(adj_joint_qd_start),
			cast<float*>(adj_joint_q),
			cast<float*>(adj_joint_qd),
			cast<df::float3*>(adj_joint_axis),
			cast<float*>(adj_joint_target_ke),
			cast<float*>(adj_joint_target_kd),
			cast<spatial_matrix*>(adj_body_I_m),
			cast<spatial_transform*>(adj_body_X_sc),
			cast<spatial_transform*>(adj_body_X_sm),
			cast<spatial_transform*>(adj_joint_X_pj),
			cast<df::float3*>(adj_gravity),
			cast<spatial_vector*>(adj_joint_S_s),
			cast<spatial_matrix*>(adj_body_I_s),
			cast<spatial_vector*>(adj_body_v_s),
			cast<spatial_vector*>(adj_body_f_s),
			cast<spatial_vector*>(adj_body_a_s));
    }
}



// Python entry points
void eval_rigid_id_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_body_I_m,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_gravity,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_I_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_body_f_s,
	torch::Tensor var_body_a_s);

void eval_rigid_id_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_body_I_m,
	torch::Tensor var_body_X_sc,
	torch::Tensor var_body_X_sm,
	torch::Tensor var_joint_X_pj,
	torch::Tensor var_gravity,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_I_s,
	torch::Tensor var_body_v_s,
	torch::Tensor var_body_f_s,
	torch::Tensor var_body_a_s, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_qd,
	torch::Tensor adj_joint_axis,
	torch::Tensor adj_joint_target_ke,
	torch::Tensor adj_joint_target_kd,
	torch::Tensor adj_body_I_m,
	torch::Tensor adj_body_X_sc,
	torch::Tensor adj_body_X_sm,
	torch::Tensor adj_joint_X_pj,
	torch::Tensor adj_gravity,
	torch::Tensor adj_joint_S_s,
	torch::Tensor adj_body_I_s,
	torch::Tensor adj_body_v_s,
	torch::Tensor adj_body_f_s,
	torch::Tensor adj_body_a_s);


void eval_rigid_tau_cpu_kernel_forward(
	int* var_articulation_start,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_act,
	float* var_joint_target,
	float* var_joint_target_ke,
	float* var_joint_target_kd,
	float* var_joint_limit_lower,
	float* var_joint_limit_upper,
	float* var_joint_limit_ke,
	float* var_joint_limit_kd,
	df::float3* var_joint_axis,
	spatial_vector* var_joint_S_s,
	spatial_vector* var_body_fb_s,
	spatial_vector* var_body_ft_s,
	float* var_tau)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    const int var_6 = 0;
    int var_7;
    int var_8;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    for (var_7=var_6; var_7 < var_5; ++var_7) {
    	var_8 = compute_link_tau_cpu_func(var_7, var_4, var_joint_type, var_joint_parent, var_joint_q_start, var_joint_qd_start, var_joint_q, var_joint_qd, var_joint_act, var_joint_target, var_joint_target_ke, var_joint_target_kd, var_joint_limit_lower, var_joint_limit_upper, var_joint_limit_ke, var_joint_limit_kd, var_joint_S_s, var_body_fb_s, var_body_ft_s, var_tau);
    }
    

}

void eval_rigid_tau_cpu_kernel_backward(
	int* var_articulation_start,
	int* var_joint_type,
	int* var_joint_parent,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_act,
	float* var_joint_target,
	float* var_joint_target_ke,
	float* var_joint_target_kd,
	float* var_joint_limit_lower,
	float* var_joint_limit_upper,
	float* var_joint_limit_ke,
	float* var_joint_limit_kd,
	df::float3* var_joint_axis,
	spatial_vector* var_joint_S_s,
	spatial_vector* var_body_fb_s,
	spatial_vector* var_body_ft_s,
	float* var_tau, 
	int* adj_articulation_start,
	int* adj_joint_type,
	int* adj_joint_parent,
	int* adj_joint_q_start,
	int* adj_joint_qd_start,
	float* adj_joint_q,
	float* adj_joint_qd,
	float* adj_joint_act,
	float* adj_joint_target,
	float* adj_joint_target_ke,
	float* adj_joint_target_kd,
	float* adj_joint_limit_lower,
	float* adj_joint_limit_upper,
	float* adj_joint_limit_ke,
	float* adj_joint_limit_kd,
	df::float3* adj_joint_axis,
	spatial_vector* adj_joint_S_s,
	spatial_vector* adj_body_fb_s,
	spatial_vector* adj_body_ft_s,
	float* adj_tau)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    const int var_6 = 0;
    int var_7;
    int var_8;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    if (false) {
    	var_8 = compute_link_tau_cpu_func(var_7, var_4, var_joint_type, var_joint_parent, var_joint_q_start, var_joint_qd_start, var_joint_q, var_joint_qd, var_joint_act, var_joint_target, var_joint_target_ke, var_joint_target_kd, var_joint_limit_lower, var_joint_limit_upper, var_joint_limit_ke, var_joint_limit_kd, var_joint_S_s, var_body_fb_s, var_body_ft_s, var_tau);
    }
    //---------
    // reverse
    for (var_7=var_5-1; var_7 >= var_6; --var_7) {
    	adj_compute_link_tau_cpu_func(var_7, var_4, var_joint_type, var_joint_parent, var_joint_q_start, var_joint_qd_start, var_joint_q, var_joint_qd, var_joint_act, var_joint_target, var_joint_target_ke, var_joint_target_kd, var_joint_limit_lower, var_joint_limit_upper, var_joint_limit_ke, var_joint_limit_kd, var_joint_S_s, var_body_fb_s, var_body_ft_s, var_tau, adj_7, adj_4, adj_joint_type, adj_joint_parent, adj_joint_q_start, adj_joint_qd_start, adj_joint_q, adj_joint_qd, adj_joint_act, adj_joint_target, adj_joint_target_ke, adj_joint_target_kd, adj_joint_limit_lower, adj_joint_limit_upper, adj_joint_limit_ke, adj_joint_limit_kd, adj_joint_S_s, adj_body_fb_s, adj_body_ft_s, adj_tau, adj_8);
    }
    df::adj_sub(var_4, var_1, adj_4, adj_1, adj_5);
    df::adj_load(var_articulation_start, var_3, adj_articulation_start, adj_3, adj_4);
    df::adj_add(var_0, var_2, adj_0, adj_2, adj_3);
    df::adj_load(var_articulation_start, var_0, adj_articulation_start, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_tau_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_act,
	torch::Tensor var_joint_target,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_joint_limit_lower,
	torch::Tensor var_joint_limit_upper,
	torch::Tensor var_joint_limit_ke,
	torch::Tensor var_joint_limit_kd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_fb_s,
	torch::Tensor var_body_ft_s,
	torch::Tensor var_tau)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_tau_cpu_kernel_forward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<float*>(var_joint_qd),
			cast<float*>(var_joint_act),
			cast<float*>(var_joint_target),
			cast<float*>(var_joint_target_ke),
			cast<float*>(var_joint_target_kd),
			cast<float*>(var_joint_limit_lower),
			cast<float*>(var_joint_limit_upper),
			cast<float*>(var_joint_limit_ke),
			cast<float*>(var_joint_limit_kd),
			cast<df::float3*>(var_joint_axis),
			cast<spatial_vector*>(var_joint_S_s),
			cast<spatial_vector*>(var_body_fb_s),
			cast<spatial_vector*>(var_body_ft_s),
			cast<float*>(var_tau));
    }
}

void eval_rigid_tau_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_act,
	torch::Tensor var_joint_target,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_joint_limit_lower,
	torch::Tensor var_joint_limit_upper,
	torch::Tensor var_joint_limit_ke,
	torch::Tensor var_joint_limit_kd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_fb_s,
	torch::Tensor var_body_ft_s,
	torch::Tensor var_tau, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_qd,
	torch::Tensor adj_joint_act,
	torch::Tensor adj_joint_target,
	torch::Tensor adj_joint_target_ke,
	torch::Tensor adj_joint_target_kd,
	torch::Tensor adj_joint_limit_lower,
	torch::Tensor adj_joint_limit_upper,
	torch::Tensor adj_joint_limit_ke,
	torch::Tensor adj_joint_limit_kd,
	torch::Tensor adj_joint_axis,
	torch::Tensor adj_joint_S_s,
	torch::Tensor adj_body_fb_s,
	torch::Tensor adj_body_ft_s,
	torch::Tensor adj_tau)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_tau_cpu_kernel_backward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<float*>(var_joint_qd),
			cast<float*>(var_joint_act),
			cast<float*>(var_joint_target),
			cast<float*>(var_joint_target_ke),
			cast<float*>(var_joint_target_kd),
			cast<float*>(var_joint_limit_lower),
			cast<float*>(var_joint_limit_upper),
			cast<float*>(var_joint_limit_ke),
			cast<float*>(var_joint_limit_kd),
			cast<df::float3*>(var_joint_axis),
			cast<spatial_vector*>(var_joint_S_s),
			cast<spatial_vector*>(var_body_fb_s),
			cast<spatial_vector*>(var_body_ft_s),
			cast<float*>(var_tau), 
			cast<int*>(adj_articulation_start),
			cast<int*>(adj_joint_type),
			cast<int*>(adj_joint_parent),
			cast<int*>(adj_joint_q_start),
			cast<int*>(adj_joint_qd_start),
			cast<float*>(adj_joint_q),
			cast<float*>(adj_joint_qd),
			cast<float*>(adj_joint_act),
			cast<float*>(adj_joint_target),
			cast<float*>(adj_joint_target_ke),
			cast<float*>(adj_joint_target_kd),
			cast<float*>(adj_joint_limit_lower),
			cast<float*>(adj_joint_limit_upper),
			cast<float*>(adj_joint_limit_ke),
			cast<float*>(adj_joint_limit_kd),
			cast<df::float3*>(adj_joint_axis),
			cast<spatial_vector*>(adj_joint_S_s),
			cast<spatial_vector*>(adj_body_fb_s),
			cast<spatial_vector*>(adj_body_ft_s),
			cast<float*>(adj_tau));
    }
}



// Python entry points
void eval_rigid_tau_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_act,
	torch::Tensor var_joint_target,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_joint_limit_lower,
	torch::Tensor var_joint_limit_upper,
	torch::Tensor var_joint_limit_ke,
	torch::Tensor var_joint_limit_kd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_fb_s,
	torch::Tensor var_body_ft_s,
	torch::Tensor var_tau);

void eval_rigid_tau_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_act,
	torch::Tensor var_joint_target,
	torch::Tensor var_joint_target_ke,
	torch::Tensor var_joint_target_kd,
	torch::Tensor var_joint_limit_lower,
	torch::Tensor var_joint_limit_upper,
	torch::Tensor var_joint_limit_ke,
	torch::Tensor var_joint_limit_kd,
	torch::Tensor var_joint_axis,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_body_fb_s,
	torch::Tensor var_body_ft_s,
	torch::Tensor var_tau, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_qd,
	torch::Tensor adj_joint_act,
	torch::Tensor adj_joint_target,
	torch::Tensor adj_joint_target_ke,
	torch::Tensor adj_joint_target_kd,
	torch::Tensor adj_joint_limit_lower,
	torch::Tensor adj_joint_limit_upper,
	torch::Tensor adj_joint_limit_ke,
	torch::Tensor adj_joint_limit_kd,
	torch::Tensor adj_joint_axis,
	torch::Tensor adj_joint_S_s,
	torch::Tensor adj_body_fb_s,
	torch::Tensor adj_body_ft_s,
	torch::Tensor adj_tau);


void eval_rigid_jacobian_cpu_kernel_forward(
	int* var_articulation_start,
	int* var_articulation_J_start,
	int* var_joint_parent,
	int* var_joint_qd_start,
	spatial_vector* var_joint_S_s,
	float* var_J)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    var_6 = df::load(var_articulation_J_start, var_0);
    df::spatial_jacobian(var_joint_S_s, var_joint_parent, var_joint_qd_start, var_1, var_5, var_6, var_J);
    

}

void eval_rigid_jacobian_cpu_kernel_backward(
	int* var_articulation_start,
	int* var_articulation_J_start,
	int* var_joint_parent,
	int* var_joint_qd_start,
	spatial_vector* var_joint_S_s,
	float* var_J, 
	int* adj_articulation_start,
	int* adj_articulation_J_start,
	int* adj_joint_parent,
	int* adj_joint_qd_start,
	spatial_vector* adj_joint_S_s,
	float* adj_J)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    var_6 = df::load(var_articulation_J_start, var_0);
    df::spatial_jacobian(var_joint_S_s, var_joint_parent, var_joint_qd_start, var_1, var_5, var_6, var_J);
    //---------
    // reverse
    df::adj_spatial_jacobian(var_joint_S_s, var_joint_parent, var_joint_qd_start, var_1, var_5, var_6, var_J, adj_joint_S_s, adj_joint_parent, adj_joint_qd_start, adj_1, adj_5, adj_6, adj_J);
    df::adj_load(var_articulation_J_start, var_0, adj_articulation_J_start, adj_0, adj_6);
    df::adj_sub(var_4, var_1, adj_4, adj_1, adj_5);
    df::adj_load(var_articulation_start, var_3, adj_articulation_start, adj_3, adj_4);
    df::adj_add(var_0, var_2, adj_0, adj_2, adj_3);
    df::adj_load(var_articulation_start, var_0, adj_articulation_start, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_jacobian_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_J_start,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_J)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_jacobian_cpu_kernel_forward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_articulation_J_start),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_qd_start),
			cast<spatial_vector*>(var_joint_S_s),
			cast<float*>(var_J));
    }
}

void eval_rigid_jacobian_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_J_start,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_J, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_articulation_J_start,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_S_s,
	torch::Tensor adj_J)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_jacobian_cpu_kernel_backward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_articulation_J_start),
			cast<int*>(var_joint_parent),
			cast<int*>(var_joint_qd_start),
			cast<spatial_vector*>(var_joint_S_s),
			cast<float*>(var_J), 
			cast<int*>(adj_articulation_start),
			cast<int*>(adj_articulation_J_start),
			cast<int*>(adj_joint_parent),
			cast<int*>(adj_joint_qd_start),
			cast<spatial_vector*>(adj_joint_S_s),
			cast<float*>(adj_J));
    }
}



// Python entry points
void eval_rigid_jacobian_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_J_start,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_J);

void eval_rigid_jacobian_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_J_start,
	torch::Tensor var_joint_parent,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_S_s,
	torch::Tensor var_J, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_articulation_J_start,
	torch::Tensor adj_joint_parent,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_S_s,
	torch::Tensor adj_J);


void eval_rigid_mass_cpu_kernel_forward(
	int* var_articulation_start,
	int* var_articulation_M_start,
	spatial_matrix* var_body_I_s,
	float* var_M)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    var_6 = df::load(var_articulation_M_start, var_0);
    df::spatial_mass(var_body_I_s, var_1, var_5, var_6, var_M);
    

}

void eval_rigid_mass_cpu_kernel_backward(
	int* var_articulation_start,
	int* var_articulation_M_start,
	spatial_matrix* var_body_I_s,
	float* var_M, 
	int* adj_articulation_start,
	int* adj_articulation_M_start,
	spatial_matrix* adj_body_I_s,
	float* adj_M)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    const int var_2 = 1;
    int var_3;
    int var_4;
    int var_5;
    int var_6;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_articulation_start, var_0);
    var_3 = df::add(var_0, var_2);
    var_4 = df::load(var_articulation_start, var_3);
    var_5 = df::sub(var_4, var_1);
    var_6 = df::load(var_articulation_M_start, var_0);
    df::spatial_mass(var_body_I_s, var_1, var_5, var_6, var_M);
    //---------
    // reverse
    df::adj_spatial_mass(var_body_I_s, var_1, var_5, var_6, var_M, adj_body_I_s, adj_1, adj_5, adj_6, adj_M);
    df::adj_load(var_articulation_M_start, var_0, adj_articulation_M_start, adj_0, adj_6);
    df::adj_sub(var_4, var_1, adj_4, adj_1, adj_5);
    df::adj_load(var_articulation_start, var_3, adj_articulation_start, adj_3, adj_4);
    df::adj_add(var_0, var_2, adj_0, adj_2, adj_3);
    df::adj_load(var_articulation_start, var_0, adj_articulation_start, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_mass_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_M_start,
	torch::Tensor var_body_I_s,
	torch::Tensor var_M)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_mass_cpu_kernel_forward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_articulation_M_start),
			cast<spatial_matrix*>(var_body_I_s),
			cast<float*>(var_M));
    }
}

void eval_rigid_mass_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_M_start,
	torch::Tensor var_body_I_s,
	torch::Tensor var_M, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_articulation_M_start,
	torch::Tensor adj_body_I_s,
	torch::Tensor adj_M)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_mass_cpu_kernel_backward(
			cast<int*>(var_articulation_start),
			cast<int*>(var_articulation_M_start),
			cast<spatial_matrix*>(var_body_I_s),
			cast<float*>(var_M), 
			cast<int*>(adj_articulation_start),
			cast<int*>(adj_articulation_M_start),
			cast<spatial_matrix*>(adj_body_I_s),
			cast<float*>(adj_M));
    }
}



// Python entry points
void eval_rigid_mass_cpu_forward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_M_start,
	torch::Tensor var_body_I_s,
	torch::Tensor var_M);

void eval_rigid_mass_cpu_backward(int dim, 
	torch::Tensor var_articulation_start,
	torch::Tensor var_articulation_M_start,
	torch::Tensor var_body_I_s,
	torch::Tensor var_M, 
	torch::Tensor adj_articulation_start,
	torch::Tensor adj_articulation_M_start,
	torch::Tensor adj_body_I_s,
	torch::Tensor adj_M);


void eval_dense_gemm_cpu_kernel_forward(
	int var_m,
	int var_n,
	int var_p,
	int var_t1,
	int var_t2,
	float* var_A,
	float* var_B,
	float* var_C)
{
        //---------
    // primal vars
    //---------
    // forward
    df::dense_gemm(var_m, var_n, var_p, var_t1, var_t2, var_A, var_B, var_C);
    

}

void eval_dense_gemm_cpu_kernel_backward(
	int var_m,
	int var_n,
	int var_p,
	int var_t1,
	int var_t2,
	float* var_A,
	float* var_B,
	float* var_C, 
	int adj_m,
	int adj_n,
	int adj_p,
	int adj_t1,
	int adj_t2,
	float* adj_A,
	float* adj_B,
	float* adj_C)
{
        //---------
    // primal vars
    //---------
    // dual vars
    //---------
    // forward
    df::dense_gemm(var_m, var_n, var_p, var_t1, var_t2, var_A, var_B, var_C);
    //---------
    // reverse
    df::adj_dense_gemm(var_m, var_n, var_p, var_t1, var_t2, var_A, var_B, var_C, adj_m, adj_n, adj_p, adj_t1, adj_t2, adj_A, adj_B, adj_C);
    return;

}



// Python entry points
void eval_dense_gemm_cpu_forward(int dim, 
	int var_m,
	int var_n,
	int var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_gemm_cpu_kernel_forward(
			var_m,
			var_n,
			var_p,
			var_t1,
			var_t2,
			cast<float*>(var_A),
			cast<float*>(var_B),
			cast<float*>(var_C));
    }
}

void eval_dense_gemm_cpu_backward(int dim, 
	int var_m,
	int var_n,
	int var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C, 
	int adj_m,
	int adj_n,
	int adj_p,
	int adj_t1,
	int adj_t2,
	torch::Tensor adj_A,
	torch::Tensor adj_B,
	torch::Tensor adj_C)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_gemm_cpu_kernel_backward(
			var_m,
			var_n,
			var_p,
			var_t1,
			var_t2,
			cast<float*>(var_A),
			cast<float*>(var_B),
			cast<float*>(var_C), 
			adj_m,
			adj_n,
			adj_p,
			adj_t1,
			adj_t2,
			cast<float*>(adj_A),
			cast<float*>(adj_B),
			cast<float*>(adj_C));
    }
}



// Python entry points
void eval_dense_gemm_cpu_forward(int dim, 
	int var_m,
	int var_n,
	int var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C);

void eval_dense_gemm_cpu_backward(int dim, 
	int var_m,
	int var_n,
	int var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C, 
	int adj_m,
	int adj_n,
	int adj_p,
	int adj_t1,
	int adj_t2,
	torch::Tensor adj_A,
	torch::Tensor adj_B,
	torch::Tensor adj_C);


void eval_dense_gemm_batched_cpu_kernel_forward(
	int* var_m,
	int* var_n,
	int* var_p,
	int var_t1,
	int var_t2,
	int* var_A_start,
	int* var_B_start,
	int* var_C_start,
	float* var_A,
	float* var_B,
	float* var_C)
{
        //---------
    // primal vars
    //---------
    // forward
    df::dense_gemm_batched(var_m, var_n, var_p, var_t1, var_t2, var_A_start, var_B_start, var_C_start, var_A, var_B, var_C);
    

}

void eval_dense_gemm_batched_cpu_kernel_backward(
	int* var_m,
	int* var_n,
	int* var_p,
	int var_t1,
	int var_t2,
	int* var_A_start,
	int* var_B_start,
	int* var_C_start,
	float* var_A,
	float* var_B,
	float* var_C, 
	int* adj_m,
	int* adj_n,
	int* adj_p,
	int adj_t1,
	int adj_t2,
	int* adj_A_start,
	int* adj_B_start,
	int* adj_C_start,
	float* adj_A,
	float* adj_B,
	float* adj_C)
{
        //---------
    // primal vars
    //---------
    // dual vars
    //---------
    // forward
    df::dense_gemm_batched(var_m, var_n, var_p, var_t1, var_t2, var_A_start, var_B_start, var_C_start, var_A, var_B, var_C);
    //---------
    // reverse
    df::adj_dense_gemm_batched(var_m, var_n, var_p, var_t1, var_t2, var_A_start, var_B_start, var_C_start, var_A, var_B, var_C, adj_m, adj_n, adj_p, adj_t1, adj_t2, adj_A_start, adj_B_start, adj_C_start, adj_A, adj_B, adj_C);
    return;

}



// Python entry points
void eval_dense_gemm_batched_cpu_forward(int dim, 
	torch::Tensor var_m,
	torch::Tensor var_n,
	torch::Tensor var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A_start,
	torch::Tensor var_B_start,
	torch::Tensor var_C_start,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_gemm_batched_cpu_kernel_forward(
			cast<int*>(var_m),
			cast<int*>(var_n),
			cast<int*>(var_p),
			var_t1,
			var_t2,
			cast<int*>(var_A_start),
			cast<int*>(var_B_start),
			cast<int*>(var_C_start),
			cast<float*>(var_A),
			cast<float*>(var_B),
			cast<float*>(var_C));
    }
}

void eval_dense_gemm_batched_cpu_backward(int dim, 
	torch::Tensor var_m,
	torch::Tensor var_n,
	torch::Tensor var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A_start,
	torch::Tensor var_B_start,
	torch::Tensor var_C_start,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C, 
	torch::Tensor adj_m,
	torch::Tensor adj_n,
	torch::Tensor adj_p,
	int adj_t1,
	int adj_t2,
	torch::Tensor adj_A_start,
	torch::Tensor adj_B_start,
	torch::Tensor adj_C_start,
	torch::Tensor adj_A,
	torch::Tensor adj_B,
	torch::Tensor adj_C)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_gemm_batched_cpu_kernel_backward(
			cast<int*>(var_m),
			cast<int*>(var_n),
			cast<int*>(var_p),
			var_t1,
			var_t2,
			cast<int*>(var_A_start),
			cast<int*>(var_B_start),
			cast<int*>(var_C_start),
			cast<float*>(var_A),
			cast<float*>(var_B),
			cast<float*>(var_C), 
			cast<int*>(adj_m),
			cast<int*>(adj_n),
			cast<int*>(adj_p),
			adj_t1,
			adj_t2,
			cast<int*>(adj_A_start),
			cast<int*>(adj_B_start),
			cast<int*>(adj_C_start),
			cast<float*>(adj_A),
			cast<float*>(adj_B),
			cast<float*>(adj_C));
    }
}



// Python entry points
void eval_dense_gemm_batched_cpu_forward(int dim, 
	torch::Tensor var_m,
	torch::Tensor var_n,
	torch::Tensor var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A_start,
	torch::Tensor var_B_start,
	torch::Tensor var_C_start,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C);

void eval_dense_gemm_batched_cpu_backward(int dim, 
	torch::Tensor var_m,
	torch::Tensor var_n,
	torch::Tensor var_p,
	int var_t1,
	int var_t2,
	torch::Tensor var_A_start,
	torch::Tensor var_B_start,
	torch::Tensor var_C_start,
	torch::Tensor var_A,
	torch::Tensor var_B,
	torch::Tensor var_C, 
	torch::Tensor adj_m,
	torch::Tensor adj_n,
	torch::Tensor adj_p,
	int adj_t1,
	int adj_t2,
	torch::Tensor adj_A_start,
	torch::Tensor adj_B_start,
	torch::Tensor adj_C_start,
	torch::Tensor adj_A,
	torch::Tensor adj_B,
	torch::Tensor adj_C);


void eval_dense_cholesky_cpu_kernel_forward(
	int var_n,
	float* var_A,
	float* var_regularization,
	float* var_L)
{
        //---------
    // primal vars
    //---------
    // forward
    df::dense_chol(var_n, var_A, var_regularization, var_L);
    

}

void eval_dense_cholesky_cpu_kernel_backward(
	int var_n,
	float* var_A,
	float* var_regularization,
	float* var_L, 
	int adj_n,
	float* adj_A,
	float* adj_regularization,
	float* adj_L)
{
        //---------
    // primal vars
    //---------
    // dual vars
    //---------
    // forward
    df::dense_chol(var_n, var_A, var_regularization, var_L);
    //---------
    // reverse
    df::adj_dense_chol(var_n, var_A, var_regularization, var_L, adj_n, adj_A, adj_regularization, adj_L);
    return;

}



// Python entry points
void eval_dense_cholesky_cpu_forward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_cholesky_cpu_kernel_forward(
			var_n,
			cast<float*>(var_A),
			cast<float*>(var_regularization),
			cast<float*>(var_L));
    }
}

void eval_dense_cholesky_cpu_backward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L, 
	int adj_n,
	torch::Tensor adj_A,
	torch::Tensor adj_regularization,
	torch::Tensor adj_L)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_cholesky_cpu_kernel_backward(
			var_n,
			cast<float*>(var_A),
			cast<float*>(var_regularization),
			cast<float*>(var_L), 
			adj_n,
			cast<float*>(adj_A),
			cast<float*>(adj_regularization),
			cast<float*>(adj_L));
    }
}



// Python entry points
void eval_dense_cholesky_cpu_forward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L);

void eval_dense_cholesky_cpu_backward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L, 
	int adj_n,
	torch::Tensor adj_A,
	torch::Tensor adj_regularization,
	torch::Tensor adj_L);


void eval_dense_cholesky_batched_cpu_kernel_forward(
	int* var_A_start,
	int* var_A_dim,
	float* var_A,
	float* var_regularization,
	float* var_L)
{
        //---------
    // primal vars
    //---------
    // forward
    df::dense_chol_batched(var_A_start, var_A_dim, var_A, var_regularization, var_L);
    

}

void eval_dense_cholesky_batched_cpu_kernel_backward(
	int* var_A_start,
	int* var_A_dim,
	float* var_A,
	float* var_regularization,
	float* var_L, 
	int* adj_A_start,
	int* adj_A_dim,
	float* adj_A,
	float* adj_regularization,
	float* adj_L)
{
        //---------
    // primal vars
    //---------
    // dual vars
    //---------
    // forward
    df::dense_chol_batched(var_A_start, var_A_dim, var_A, var_regularization, var_L);
    //---------
    // reverse
    df::adj_dense_chol_batched(var_A_start, var_A_dim, var_A, var_regularization, var_L, adj_A_start, adj_A_dim, adj_A, adj_regularization, adj_L);
    return;

}



// Python entry points
void eval_dense_cholesky_batched_cpu_forward(int dim, 
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_cholesky_batched_cpu_kernel_forward(
			cast<int*>(var_A_start),
			cast<int*>(var_A_dim),
			cast<float*>(var_A),
			cast<float*>(var_regularization),
			cast<float*>(var_L));
    }
}

void eval_dense_cholesky_batched_cpu_backward(int dim, 
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L, 
	torch::Tensor adj_A_start,
	torch::Tensor adj_A_dim,
	torch::Tensor adj_A,
	torch::Tensor adj_regularization,
	torch::Tensor adj_L)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_cholesky_batched_cpu_kernel_backward(
			cast<int*>(var_A_start),
			cast<int*>(var_A_dim),
			cast<float*>(var_A),
			cast<float*>(var_regularization),
			cast<float*>(var_L), 
			cast<int*>(adj_A_start),
			cast<int*>(adj_A_dim),
			cast<float*>(adj_A),
			cast<float*>(adj_regularization),
			cast<float*>(adj_L));
    }
}



// Python entry points
void eval_dense_cholesky_batched_cpu_forward(int dim, 
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L);

void eval_dense_cholesky_batched_cpu_backward(int dim, 
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_regularization,
	torch::Tensor var_L, 
	torch::Tensor adj_A_start,
	torch::Tensor adj_A_dim,
	torch::Tensor adj_A,
	torch::Tensor adj_regularization,
	torch::Tensor adj_L);


void eval_dense_subs_cpu_kernel_forward(
	int var_n,
	float* var_L,
	float* var_b,
	float* var_x)
{
        //---------
    // primal vars
    //---------
    // forward
    df::dense_subs(var_n, var_L, var_b, var_x);
    

}

void eval_dense_subs_cpu_kernel_backward(
	int var_n,
	float* var_L,
	float* var_b,
	float* var_x, 
	int adj_n,
	float* adj_L,
	float* adj_b,
	float* adj_x)
{
        //---------
    // primal vars
    //---------
    // dual vars
    //---------
    // forward
    df::dense_subs(var_n, var_L, var_b, var_x);
    //---------
    // reverse
    df::adj_dense_subs(var_n, var_L, var_b, var_x, adj_n, adj_L, adj_b, adj_x);
    return;

}



// Python entry points
void eval_dense_subs_cpu_forward(int dim, 
	int var_n,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_x)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_subs_cpu_kernel_forward(
			var_n,
			cast<float*>(var_L),
			cast<float*>(var_b),
			cast<float*>(var_x));
    }
}

void eval_dense_subs_cpu_backward(int dim, 
	int var_n,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_x, 
	int adj_n,
	torch::Tensor adj_L,
	torch::Tensor adj_b,
	torch::Tensor adj_x)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_subs_cpu_kernel_backward(
			var_n,
			cast<float*>(var_L),
			cast<float*>(var_b),
			cast<float*>(var_x), 
			adj_n,
			cast<float*>(adj_L),
			cast<float*>(adj_b),
			cast<float*>(adj_x));
    }
}



// Python entry points
void eval_dense_subs_cpu_forward(int dim, 
	int var_n,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_x);

void eval_dense_subs_cpu_backward(int dim, 
	int var_n,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_x, 
	int adj_n,
	torch::Tensor adj_L,
	torch::Tensor adj_b,
	torch::Tensor adj_x);


void eval_dense_solve_cpu_kernel_forward(
	int var_n,
	float* var_A,
	float* var_L,
	float* var_b,
	float* var_tmp,
	float* var_x)
{
        //---------
    // primal vars
    //---------
    // forward
    df::dense_solve(var_n, var_A, var_L, var_b, var_tmp, var_x);
    

}

void eval_dense_solve_cpu_kernel_backward(
	int var_n,
	float* var_A,
	float* var_L,
	float* var_b,
	float* var_tmp,
	float* var_x, 
	int adj_n,
	float* adj_A,
	float* adj_L,
	float* adj_b,
	float* adj_tmp,
	float* adj_x)
{
        //---------
    // primal vars
    //---------
    // dual vars
    //---------
    // forward
    df::dense_solve(var_n, var_A, var_L, var_b, var_tmp, var_x);
    //---------
    // reverse
    df::adj_dense_solve(var_n, var_A, var_L, var_b, var_tmp, var_x, adj_n, adj_A, adj_L, adj_b, adj_tmp, adj_x);
    return;

}



// Python entry points
void eval_dense_solve_cpu_forward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_solve_cpu_kernel_forward(
			var_n,
			cast<float*>(var_A),
			cast<float*>(var_L),
			cast<float*>(var_b),
			cast<float*>(var_tmp),
			cast<float*>(var_x));
    }
}

void eval_dense_solve_cpu_backward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x, 
	int adj_n,
	torch::Tensor adj_A,
	torch::Tensor adj_L,
	torch::Tensor adj_b,
	torch::Tensor adj_tmp,
	torch::Tensor adj_x)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_solve_cpu_kernel_backward(
			var_n,
			cast<float*>(var_A),
			cast<float*>(var_L),
			cast<float*>(var_b),
			cast<float*>(var_tmp),
			cast<float*>(var_x), 
			adj_n,
			cast<float*>(adj_A),
			cast<float*>(adj_L),
			cast<float*>(adj_b),
			cast<float*>(adj_tmp),
			cast<float*>(adj_x));
    }
}



// Python entry points
void eval_dense_solve_cpu_forward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x);

void eval_dense_solve_cpu_backward(int dim, 
	int var_n,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x, 
	int adj_n,
	torch::Tensor adj_A,
	torch::Tensor adj_L,
	torch::Tensor adj_b,
	torch::Tensor adj_tmp,
	torch::Tensor adj_x);


void eval_dense_solve_batched_cpu_kernel_forward(
	int* var_b_start,
	int* var_A_start,
	int* var_A_dim,
	float* var_A,
	float* var_L,
	float* var_b,
	float* var_tmp,
	float* var_x)
{
        //---------
    // primal vars
    //---------
    // forward
    df::dense_solve_batched(var_b_start, var_A_start, var_A_dim, var_A, var_L, var_b, var_tmp, var_x);
    

}

void eval_dense_solve_batched_cpu_kernel_backward(
	int* var_b_start,
	int* var_A_start,
	int* var_A_dim,
	float* var_A,
	float* var_L,
	float* var_b,
	float* var_tmp,
	float* var_x, 
	int* adj_b_start,
	int* adj_A_start,
	int* adj_A_dim,
	float* adj_A,
	float* adj_L,
	float* adj_b,
	float* adj_tmp,
	float* adj_x)
{
        //---------
    // primal vars
    //---------
    // dual vars
    //---------
    // forward
    df::dense_solve_batched(var_b_start, var_A_start, var_A_dim, var_A, var_L, var_b, var_tmp, var_x);
    //---------
    // reverse
    df::adj_dense_solve_batched(var_b_start, var_A_start, var_A_dim, var_A, var_L, var_b, var_tmp, var_x, adj_b_start, adj_A_start, adj_A_dim, adj_A, adj_L, adj_b, adj_tmp, adj_x);
    return;

}



// Python entry points
void eval_dense_solve_batched_cpu_forward(int dim, 
	torch::Tensor var_b_start,
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_solve_batched_cpu_kernel_forward(
			cast<int*>(var_b_start),
			cast<int*>(var_A_start),
			cast<int*>(var_A_dim),
			cast<float*>(var_A),
			cast<float*>(var_L),
			cast<float*>(var_b),
			cast<float*>(var_tmp),
			cast<float*>(var_x));
    }
}

void eval_dense_solve_batched_cpu_backward(int dim, 
	torch::Tensor var_b_start,
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x, 
	torch::Tensor adj_b_start,
	torch::Tensor adj_A_start,
	torch::Tensor adj_A_dim,
	torch::Tensor adj_A,
	torch::Tensor adj_L,
	torch::Tensor adj_b,
	torch::Tensor adj_tmp,
	torch::Tensor adj_x)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_dense_solve_batched_cpu_kernel_backward(
			cast<int*>(var_b_start),
			cast<int*>(var_A_start),
			cast<int*>(var_A_dim),
			cast<float*>(var_A),
			cast<float*>(var_L),
			cast<float*>(var_b),
			cast<float*>(var_tmp),
			cast<float*>(var_x), 
			cast<int*>(adj_b_start),
			cast<int*>(adj_A_start),
			cast<int*>(adj_A_dim),
			cast<float*>(adj_A),
			cast<float*>(adj_L),
			cast<float*>(adj_b),
			cast<float*>(adj_tmp),
			cast<float*>(adj_x));
    }
}



// Python entry points
void eval_dense_solve_batched_cpu_forward(int dim, 
	torch::Tensor var_b_start,
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x);

void eval_dense_solve_batched_cpu_backward(int dim, 
	torch::Tensor var_b_start,
	torch::Tensor var_A_start,
	torch::Tensor var_A_dim,
	torch::Tensor var_A,
	torch::Tensor var_L,
	torch::Tensor var_b,
	torch::Tensor var_tmp,
	torch::Tensor var_x, 
	torch::Tensor adj_b_start,
	torch::Tensor adj_A_start,
	torch::Tensor adj_A_dim,
	torch::Tensor adj_A,
	torch::Tensor adj_L,
	torch::Tensor adj_b,
	torch::Tensor adj_tmp,
	torch::Tensor adj_x);


void eval_rigid_integrate_cpu_kernel_forward(
	int* var_joint_type,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_qdd,
	float var_dt,
	float* var_joint_q_new,
	float* var_joint_qd_new)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    int var_3;
    int var_4;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_joint_type, var_0);
    var_2 = df::load(var_joint_q_start, var_0);
    var_3 = df::load(var_joint_qd_start, var_0);
    var_4 = jcalc_integrate_cpu_func(var_1, var_joint_q, var_joint_qd, var_joint_qdd, var_2, var_3, var_dt, var_joint_q_new, var_joint_qd_new);
    

}

void eval_rigid_integrate_cpu_kernel_backward(
	int* var_joint_type,
	int* var_joint_q_start,
	int* var_joint_qd_start,
	float* var_joint_q,
	float* var_joint_qd,
	float* var_joint_qdd,
	float var_dt,
	float* var_joint_q_new,
	float* var_joint_qd_new, 
	int* adj_joint_type,
	int* adj_joint_q_start,
	int* adj_joint_qd_start,
	float* adj_joint_q,
	float* adj_joint_qd,
	float* adj_joint_qdd,
	float adj_dt,
	float* adj_joint_q_new,
	float* adj_joint_qd_new)
{
        //---------
    // primal vars
    int var_0;
    int var_1;
    int var_2;
    int var_3;
    int var_4;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_joint_type, var_0);
    var_2 = df::load(var_joint_q_start, var_0);
    var_3 = df::load(var_joint_qd_start, var_0);
    var_4 = jcalc_integrate_cpu_func(var_1, var_joint_q, var_joint_qd, var_joint_qdd, var_2, var_3, var_dt, var_joint_q_new, var_joint_qd_new);
    //---------
    // reverse
    adj_jcalc_integrate_cpu_func(var_1, var_joint_q, var_joint_qd, var_joint_qdd, var_2, var_3, var_dt, var_joint_q_new, var_joint_qd_new, adj_1, adj_joint_q, adj_joint_qd, adj_joint_qdd, adj_2, adj_3, adj_dt, adj_joint_q_new, adj_joint_qd_new, adj_4);
    df::adj_load(var_joint_qd_start, var_0, adj_joint_qd_start, adj_0, adj_3);
    df::adj_load(var_joint_q_start, var_0, adj_joint_q_start, adj_0, adj_2);
    df::adj_load(var_joint_type, var_0, adj_joint_type, adj_0, adj_1);
    return;

}



// Python entry points
void eval_rigid_integrate_cpu_forward(int dim, 
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_qdd,
	float var_dt,
	torch::Tensor var_joint_q_new,
	torch::Tensor var_joint_qd_new)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_integrate_cpu_kernel_forward(
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<float*>(var_joint_qd),
			cast<float*>(var_joint_qdd),
			var_dt,
			cast<float*>(var_joint_q_new),
			cast<float*>(var_joint_qd_new));
    }
}

void eval_rigid_integrate_cpu_backward(int dim, 
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_qdd,
	float var_dt,
	torch::Tensor var_joint_q_new,
	torch::Tensor var_joint_qd_new, 
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_qd,
	torch::Tensor adj_joint_qdd,
	float adj_dt,
	torch::Tensor adj_joint_q_new,
	torch::Tensor adj_joint_qd_new)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        eval_rigid_integrate_cpu_kernel_backward(
			cast<int*>(var_joint_type),
			cast<int*>(var_joint_q_start),
			cast<int*>(var_joint_qd_start),
			cast<float*>(var_joint_q),
			cast<float*>(var_joint_qd),
			cast<float*>(var_joint_qdd),
			var_dt,
			cast<float*>(var_joint_q_new),
			cast<float*>(var_joint_qd_new), 
			cast<int*>(adj_joint_type),
			cast<int*>(adj_joint_q_start),
			cast<int*>(adj_joint_qd_start),
			cast<float*>(adj_joint_q),
			cast<float*>(adj_joint_qd),
			cast<float*>(adj_joint_qdd),
			adj_dt,
			cast<float*>(adj_joint_q_new),
			cast<float*>(adj_joint_qd_new));
    }
}



// Python entry points
void eval_rigid_integrate_cpu_forward(int dim, 
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_qdd,
	float var_dt,
	torch::Tensor var_joint_q_new,
	torch::Tensor var_joint_qd_new);

void eval_rigid_integrate_cpu_backward(int dim, 
	torch::Tensor var_joint_type,
	torch::Tensor var_joint_q_start,
	torch::Tensor var_joint_qd_start,
	torch::Tensor var_joint_q,
	torch::Tensor var_joint_qd,
	torch::Tensor var_joint_qdd,
	float var_dt,
	torch::Tensor var_joint_q_new,
	torch::Tensor var_joint_qd_new, 
	torch::Tensor adj_joint_type,
	torch::Tensor adj_joint_q_start,
	torch::Tensor adj_joint_qd_start,
	torch::Tensor adj_joint_q,
	torch::Tensor adj_joint_qd,
	torch::Tensor adj_joint_qdd,
	float adj_dt,
	torch::Tensor adj_joint_q_new,
	torch::Tensor adj_joint_qd_new);


void solve_springs_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	float* var_invmass,
	int* var_spring_indices,
	float* var_spring_rest_lengths,
	float* var_spring_stiffness,
	float* var_spring_damping,
	float var_dt,
	df::float3* var_delta)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 2;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    float var_10;
    float var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    float var_19;
    const float var_20 = 1.0;
    float var_21;
    df::float3 var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_spring_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_spring_indices, var_8);
    var_10 = df::load(var_spring_stiffness, var_0);
    var_11 = df::load(var_spring_damping, var_0);
    var_12 = df::load(var_spring_rest_lengths, var_0);
    var_13 = df::load(var_x, var_5);
    var_14 = df::load(var_x, var_9);
    var_15 = df::load(var_v, var_5);
    var_16 = df::load(var_v, var_9);
    var_17 = df::sub(var_13, var_14);
    var_18 = df::sub(var_15, var_16);
    var_19 = df::length(var_17);
    var_21 = df::div(var_20, var_19);
    var_22 = df::mul(var_17, var_21);
    var_23 = df::sub(var_19, var_12);
    var_24 = df::dot(var_22, var_18);
    var_25 = df::load(var_invmass, var_5);
    var_26 = df::load(var_invmass, var_9);
    var_27 = df::add(var_25, var_26);
    var_28 = df::mul(var_10, var_dt);
    var_29 = df::mul(var_28, var_dt);
    var_30 = df::div(var_20, var_29);
    var_31 = df::div(var_23, var_27);
    var_32 = df::mul(var_22, var_31);
    var_33 = df::mul(var_32, var_25);
    df::atomic_sub(var_delta, var_5, var_33);
    var_34 = df::mul(var_32, var_26);
    df::atomic_add(var_delta, var_9, var_34);
    

}

void solve_springs_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	float* var_invmass,
	int* var_spring_indices,
	float* var_spring_rest_lengths,
	float* var_spring_stiffness,
	float* var_spring_damping,
	float var_dt,
	df::float3* var_delta, 
	df::float3* adj_x,
	df::float3* adj_v,
	float* adj_invmass,
	int* adj_spring_indices,
	float* adj_spring_rest_lengths,
	float* adj_spring_stiffness,
	float* adj_spring_damping,
	float adj_dt,
	df::float3* adj_delta)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 2;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    float var_10;
    float var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    df::float3 var_15;
    df::float3 var_16;
    df::float3 var_17;
    df::float3 var_18;
    float var_19;
    const float var_20 = 1.0;
    float var_21;
    df::float3 var_22;
    float var_23;
    float var_24;
    float var_25;
    float var_26;
    float var_27;
    float var_28;
    float var_29;
    float var_30;
    float var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    float adj_10 = 0;
    float adj_11 = 0;
    float adj_12 = 0;
    df::float3 adj_13 = 0;
    df::float3 adj_14 = 0;
    df::float3 adj_15 = 0;
    df::float3 adj_16 = 0;
    df::float3 adj_17 = 0;
    df::float3 adj_18 = 0;
    float adj_19 = 0;
    float adj_20 = 0;
    float adj_21 = 0;
    df::float3 adj_22 = 0;
    float adj_23 = 0;
    float adj_24 = 0;
    float adj_25 = 0;
    float adj_26 = 0;
    float adj_27 = 0;
    float adj_28 = 0;
    float adj_29 = 0;
    float adj_30 = 0;
    float adj_31 = 0;
    df::float3 adj_32 = 0;
    df::float3 adj_33 = 0;
    df::float3 adj_34 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_spring_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_spring_indices, var_8);
    var_10 = df::load(var_spring_stiffness, var_0);
    var_11 = df::load(var_spring_damping, var_0);
    var_12 = df::load(var_spring_rest_lengths, var_0);
    var_13 = df::load(var_x, var_5);
    var_14 = df::load(var_x, var_9);
    var_15 = df::load(var_v, var_5);
    var_16 = df::load(var_v, var_9);
    var_17 = df::sub(var_13, var_14);
    var_18 = df::sub(var_15, var_16);
    var_19 = df::length(var_17);
    var_21 = df::div(var_20, var_19);
    var_22 = df::mul(var_17, var_21);
    var_23 = df::sub(var_19, var_12);
    var_24 = df::dot(var_22, var_18);
    var_25 = df::load(var_invmass, var_5);
    var_26 = df::load(var_invmass, var_9);
    var_27 = df::add(var_25, var_26);
    var_28 = df::mul(var_10, var_dt);
    var_29 = df::mul(var_28, var_dt);
    var_30 = df::div(var_20, var_29);
    var_31 = df::div(var_23, var_27);
    var_32 = df::mul(var_22, var_31);
    var_33 = df::mul(var_32, var_25);
    df::atomic_sub(var_delta, var_5, var_33);
    var_34 = df::mul(var_32, var_26);
    df::atomic_add(var_delta, var_9, var_34);
    //---------
    // reverse
    df::adj_atomic_add(var_delta, var_9, var_34, adj_delta, adj_9, adj_34);
    df::adj_mul(var_32, var_26, adj_32, adj_26, adj_34);
    df::adj_atomic_sub(var_delta, var_5, var_33, adj_delta, adj_5, adj_33);
    df::adj_mul(var_32, var_25, adj_32, adj_25, adj_33);
    df::adj_mul(var_22, var_31, adj_22, adj_31, adj_32);
    df::adj_div(var_23, var_27, adj_23, adj_27, adj_31);
    df::adj_div(var_20, var_29, adj_20, adj_29, adj_30);
    df::adj_mul(var_28, var_dt, adj_28, adj_dt, adj_29);
    df::adj_mul(var_10, var_dt, adj_10, adj_dt, adj_28);
    df::adj_add(var_25, var_26, adj_25, adj_26, adj_27);
    df::adj_load(var_invmass, var_9, adj_invmass, adj_9, adj_26);
    df::adj_load(var_invmass, var_5, adj_invmass, adj_5, adj_25);
    df::adj_dot(var_22, var_18, adj_22, adj_18, adj_24);
    df::adj_sub(var_19, var_12, adj_19, adj_12, adj_23);
    df::adj_mul(var_17, var_21, adj_17, adj_21, adj_22);
    df::adj_div(var_20, var_19, adj_20, adj_19, adj_21);
    df::adj_length(var_17, adj_17, adj_19);
    df::adj_sub(var_15, var_16, adj_15, adj_16, adj_18);
    df::adj_sub(var_13, var_14, adj_13, adj_14, adj_17);
    df::adj_load(var_v, var_9, adj_v, adj_9, adj_16);
    df::adj_load(var_v, var_5, adj_v, adj_5, adj_15);
    df::adj_load(var_x, var_9, adj_x, adj_9, adj_14);
    df::adj_load(var_x, var_5, adj_x, adj_5, adj_13);
    df::adj_load(var_spring_rest_lengths, var_0, adj_spring_rest_lengths, adj_0, adj_12);
    df::adj_load(var_spring_damping, var_0, adj_spring_damping, adj_0, adj_11);
    df::adj_load(var_spring_stiffness, var_0, adj_spring_stiffness, adj_0, adj_10);
    df::adj_load(var_spring_indices, var_8, adj_spring_indices, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_6);
    df::adj_load(var_spring_indices, var_4, adj_spring_indices, adj_4, adj_5);
    df::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    return;

}



// Python entry points
void solve_springs_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_invmass,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	float var_dt,
	torch::Tensor var_delta)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        solve_springs_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<float*>(var_invmass),
			cast<int*>(var_spring_indices),
			cast<float*>(var_spring_rest_lengths),
			cast<float*>(var_spring_stiffness),
			cast<float*>(var_spring_damping),
			var_dt,
			cast<df::float3*>(var_delta));
    }
}

void solve_springs_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_invmass,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	float var_dt,
	torch::Tensor var_delta, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_invmass,
	torch::Tensor adj_spring_indices,
	torch::Tensor adj_spring_rest_lengths,
	torch::Tensor adj_spring_stiffness,
	torch::Tensor adj_spring_damping,
	float adj_dt,
	torch::Tensor adj_delta)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        solve_springs_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<float*>(var_invmass),
			cast<int*>(var_spring_indices),
			cast<float*>(var_spring_rest_lengths),
			cast<float*>(var_spring_stiffness),
			cast<float*>(var_spring_damping),
			var_dt,
			cast<df::float3*>(var_delta), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<float*>(adj_invmass),
			cast<int*>(adj_spring_indices),
			cast<float*>(adj_spring_rest_lengths),
			cast<float*>(adj_spring_stiffness),
			cast<float*>(adj_spring_damping),
			adj_dt,
			cast<df::float3*>(adj_delta));
    }
}



// Python entry points
void solve_springs_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_invmass,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	float var_dt,
	torch::Tensor var_delta);

void solve_springs_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_invmass,
	torch::Tensor var_spring_indices,
	torch::Tensor var_spring_rest_lengths,
	torch::Tensor var_spring_stiffness,
	torch::Tensor var_spring_damping,
	float var_dt,
	torch::Tensor var_delta, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_invmass,
	torch::Tensor adj_spring_indices,
	torch::Tensor adj_spring_rest_lengths,
	torch::Tensor adj_spring_stiffness,
	torch::Tensor adj_spring_damping,
	float adj_dt,
	torch::Tensor adj_delta);


void solve_tetrahedra_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	float* var_inv_mass,
	int* var_indices,
	mat33* var_pose,
	float* var_activation,
	float* var_materials,
	float var_dt,
	float var_relaxation,
	df::float3* var_delta)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 4;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    int var_14;
    const int var_15 = 3;
    int var_16;
    int var_17;
    float var_18;
    int var_19;
    int var_20;
    float var_21;
    int var_22;
    int var_23;
    float var_24;
    int var_25;
    int var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    float var_36;
    float var_37;
    float var_38;
    float var_39;
    df::float3 var_40;
    df::float3 var_41;
    df::float3 var_42;
    df::float3 var_43;
    df::float3 var_44;
    df::float3 var_45;
    mat33 var_46;
    mat33 var_47;
    float var_48;
    const float var_49 = 6.0;
    float var_50;
    const float var_51 = 1.0;
    float var_52;
    mat33 var_53;
    float var_54;
    float var_55;
    float var_56;
    df::float3 var_57;
    float var_58;
    float var_59;
    float var_60;
    df::float3 var_61;
    float var_62;
    float var_63;
    float var_64;
    df::float3 var_65;
    float var_66;
    float var_67;
    float var_68;
    float var_69;
    float var_70;
    const float var_71 = 3.0;
    float var_72;
    float var_73;
    float var_74;
    const float var_75 = 0.0;
    bool var_76;
    bool var_77;
    float var_78;
    float var_79;
    mat33 var_80;
    mat33 var_81;
    float var_82;
    mat33 var_83;
    float var_84;
    float var_85;
    float var_86;
    float var_87;
    float var_88;
    df::float3 var_89;
    float var_90;
    float var_91;
    float var_92;
    df::float3 var_93;
    float var_94;
    float var_95;
    float var_96;
    df::float3 var_97;
    df::float3 var_98;
    df::float3 var_99;
    float var_100;
    df::float3 var_101;
    float var_102;
    float var_103;
    float var_104;
    float var_105;
    float var_106;
    float var_107;
    float var_108;
    float var_109;
    float var_110;
    float var_111;
    float var_112;
    float var_113;
    float var_114;
    float var_115;
    float var_116;
    float var_117;
    float var_118;
    df::float3 var_119;
    df::float3 var_120;
    df::float3 var_121;
    df::float3 var_122;
    float var_123;
    float var_124;
    float var_125;
    df::float3 var_126;
    df::float3 var_127;
    df::float3 var_128;
    df::float3 var_129;
    df::float3 var_130;
    df::float3 var_131;
    df::float3 var_132;
    df::float3 var_133;
    float var_134;
    df::float3 var_135;
    float var_136;
    float var_137;
    float var_138;
    float var_139;
    float var_140;
    float var_141;
    float var_142;
    float var_143;
    float var_144;
    float var_145;
    float var_146;
    float var_147;
    float var_148;
    float var_149;
    float var_150;
    float var_151;
    float var_152;
    df::float3 var_153;
    df::float3 var_154;
    df::float3 var_155;
    df::float3 var_156;
    df::float3 var_157;
    df::float3 var_158;
    df::float3 var_159;
    df::float3 var_160;
    df::float3 var_161;
    df::float3 var_162;
    df::float3 var_163;
    df::float3 var_164;
    df::float3 var_165;
    df::float3 var_166;
    df::float3 var_167;
    df::float3 var_168;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::mul(var_0, var_1);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_indices, var_16);
    var_18 = df::load(var_activation, var_0);
    var_19 = df::mul(var_0, var_15);
    var_20 = df::add(var_19, var_3);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::mul(var_0, var_15);
    var_23 = df::add(var_22, var_7);
    var_24 = df::load(var_materials, var_23);
    var_25 = df::mul(var_0, var_15);
    var_26 = df::add(var_25, var_11);
    var_27 = df::load(var_materials, var_26);
    var_28 = df::load(var_x, var_5);
    var_29 = df::load(var_x, var_9);
    var_30 = df::load(var_x, var_13);
    var_31 = df::load(var_x, var_17);
    var_32 = df::load(var_v, var_5);
    var_33 = df::load(var_v, var_9);
    var_34 = df::load(var_v, var_13);
    var_35 = df::load(var_v, var_17);
    var_36 = df::load(var_inv_mass, var_5);
    var_37 = df::load(var_inv_mass, var_9);
    var_38 = df::load(var_inv_mass, var_13);
    var_39 = df::load(var_inv_mass, var_17);
    var_40 = df::sub(var_29, var_28);
    var_41 = df::sub(var_30, var_28);
    var_42 = df::sub(var_31, var_28);
    var_43 = df::sub(var_33, var_32);
    var_44 = df::sub(var_34, var_32);
    var_45 = df::sub(var_35, var_32);
    var_46 = df::mat33(var_40, var_41, var_42);
    var_47 = df::load(var_pose, var_0);
    var_48 = df::determinant(var_47);
    var_50 = df::mul(var_48, var_49);
    var_52 = df::div(var_51, var_50);
    var_53 = df::mul(var_46, var_47);
    var_54 = df::index(var_53, var_3, var_3);
    var_55 = df::index(var_53, var_7, var_3);
    var_56 = df::index(var_53, var_11, var_3);
    var_57 = df::float3(var_54, var_55, var_56);
    var_58 = df::index(var_53, var_3, var_7);
    var_59 = df::index(var_53, var_7, var_7);
    var_60 = df::index(var_53, var_11, var_7);
    var_61 = df::float3(var_58, var_59, var_60);
    var_62 = df::index(var_53, var_3, var_11);
    var_63 = df::index(var_53, var_7, var_11);
    var_64 = df::index(var_53, var_11, var_11);
    var_65 = df::float3(var_62, var_63, var_64);
    var_66 = df::dot(var_57, var_57);
    var_67 = df::dot(var_61, var_61);
    var_68 = df::add(var_66, var_67);
    var_69 = df::dot(var_65, var_65);
    var_70 = df::add(var_68, var_69);
    var_72 = df::sub(var_70, var_71);
    var_73 = df::abs(var_72);
    var_74 = df::sqrt(var_73);
    var_76 = (var_74 == var_75);
    if (var_76) {
    	return;
    }
    var_77 = (var_70 < var_71);
    if (var_77) {
    	var_78 = df::sub(var_75, var_74);
    }
    var_79 = df::select(var_77, var_74, var_78);
    var_80 = df::transpose(var_47);
    var_81 = df::mul(var_53, var_80);
    var_82 = df::div(var_51, var_79);
    var_83 = df::mul(var_81, var_82);
    var_84 = df::div(var_21, var_24);
    var_85 = df::add(var_51, var_84);
    var_86 = df::index(var_83, var_3, var_3);
    var_87 = df::index(var_83, var_7, var_3);
    var_88 = df::index(var_83, var_11, var_3);
    var_89 = df::float3(var_86, var_87, var_88);
    var_90 = df::index(var_83, var_3, var_7);
    var_91 = df::index(var_83, var_7, var_7);
    var_92 = df::index(var_83, var_11, var_7);
    var_93 = df::float3(var_90, var_91, var_92);
    var_94 = df::index(var_83, var_3, var_11);
    var_95 = df::index(var_83, var_7, var_11);
    var_96 = df::index(var_83, var_11, var_11);
    var_97 = df::float3(var_94, var_95, var_96);
    var_98 = df::add(var_89, var_93);
    var_99 = df::add(var_98, var_97);
    var_100 = df::sub(var_75, var_51);
    var_101 = df::mul(var_99, var_100);
    var_102 = df::dot(var_101, var_101);
    var_103 = df::mul(var_102, var_36);
    var_104 = df::dot(var_89, var_89);
    var_105 = df::mul(var_104, var_37);
    var_106 = df::add(var_103, var_105);
    var_107 = df::dot(var_93, var_93);
    var_108 = df::mul(var_107, var_38);
    var_109 = df::add(var_106, var_108);
    var_110 = df::dot(var_97, var_97);
    var_111 = df::mul(var_110, var_39);
    var_112 = df::add(var_109, var_111);
    var_113 = df::mul(var_21, var_dt);
    var_114 = df::mul(var_113, var_dt);
    var_115 = df::mul(var_114, var_52);
    var_116 = df::div(var_51, var_115);
    var_117 = df::add(var_112, var_116);
    var_118 = df::div(var_74, var_117);
    var_119 = df::mul(var_101, var_118);
    var_120 = df::mul(var_89, var_118);
    var_121 = df::mul(var_93, var_118);
    var_122 = df::mul(var_97, var_118);
    var_123 = df::determinant(var_53);
    var_124 = df::sub(var_123, var_85);
    var_125 = df::div(var_50, var_49);
    var_126 = df::cross(var_41, var_42);
    var_127 = df::mul(var_126, var_125);
    var_128 = df::cross(var_42, var_40);
    var_129 = df::mul(var_128, var_125);
    var_130 = df::cross(var_40, var_41);
    var_131 = df::mul(var_130, var_125);
    var_132 = df::add(var_127, var_129);
    var_133 = df::add(var_132, var_131);
    var_134 = df::sub(var_75, var_51);
    var_135 = df::mul(var_133, var_134);
    var_136 = df::dot(var_135, var_135);
    var_137 = df::mul(var_136, var_36);
    var_138 = df::dot(var_127, var_127);
    var_139 = df::mul(var_138, var_37);
    var_140 = df::add(var_137, var_139);
    var_141 = df::dot(var_129, var_129);
    var_142 = df::mul(var_141, var_38);
    var_143 = df::add(var_140, var_142);
    var_144 = df::dot(var_131, var_131);
    var_145 = df::mul(var_144, var_39);
    var_146 = df::add(var_143, var_145);
    var_147 = df::mul(var_24, var_dt);
    var_148 = df::mul(var_147, var_dt);
    var_149 = df::mul(var_148, var_52);
    var_150 = df::div(var_51, var_149);
    var_151 = df::add(var_146, var_150);
    var_152 = df::div(var_124, var_151);
    var_153 = df::mul(var_135, var_152);
    var_154 = df::add(var_119, var_153);
    var_155 = df::mul(var_127, var_152);
    var_156 = df::add(var_120, var_155);
    var_157 = df::mul(var_129, var_152);
    var_158 = df::add(var_121, var_157);
    var_159 = df::mul(var_131, var_152);
    var_160 = df::add(var_122, var_159);
    var_161 = df::mul(var_154, var_36);
    var_162 = df::mul(var_161, var_relaxation);
    df::atomic_sub(var_delta, var_5, var_162);
    var_163 = df::mul(var_156, var_37);
    var_164 = df::mul(var_163, var_relaxation);
    df::atomic_sub(var_delta, var_9, var_164);
    var_165 = df::mul(var_158, var_38);
    var_166 = df::mul(var_165, var_relaxation);
    df::atomic_sub(var_delta, var_13, var_166);
    var_167 = df::mul(var_160, var_39);
    var_168 = df::mul(var_167, var_relaxation);
    df::atomic_sub(var_delta, var_17, var_168);
    

}

void solve_tetrahedra_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	float* var_inv_mass,
	int* var_indices,
	mat33* var_pose,
	float* var_activation,
	float* var_materials,
	float var_dt,
	float var_relaxation,
	df::float3* var_delta, 
	df::float3* adj_x,
	df::float3* adj_v,
	float* adj_inv_mass,
	int* adj_indices,
	mat33* adj_pose,
	float* adj_activation,
	float* adj_materials,
	float adj_dt,
	float adj_relaxation,
	df::float3* adj_delta)
{
        //---------
    // primal vars
    int var_0;
    const int var_1 = 4;
    int var_2;
    const int var_3 = 0;
    int var_4;
    int var_5;
    int var_6;
    const int var_7 = 1;
    int var_8;
    int var_9;
    int var_10;
    const int var_11 = 2;
    int var_12;
    int var_13;
    int var_14;
    const int var_15 = 3;
    int var_16;
    int var_17;
    float var_18;
    int var_19;
    int var_20;
    float var_21;
    int var_22;
    int var_23;
    float var_24;
    int var_25;
    int var_26;
    float var_27;
    df::float3 var_28;
    df::float3 var_29;
    df::float3 var_30;
    df::float3 var_31;
    df::float3 var_32;
    df::float3 var_33;
    df::float3 var_34;
    df::float3 var_35;
    float var_36;
    float var_37;
    float var_38;
    float var_39;
    df::float3 var_40;
    df::float3 var_41;
    df::float3 var_42;
    df::float3 var_43;
    df::float3 var_44;
    df::float3 var_45;
    mat33 var_46;
    mat33 var_47;
    float var_48;
    const float var_49 = 6.0;
    float var_50;
    const float var_51 = 1.0;
    float var_52;
    mat33 var_53;
    float var_54;
    float var_55;
    float var_56;
    df::float3 var_57;
    float var_58;
    float var_59;
    float var_60;
    df::float3 var_61;
    float var_62;
    float var_63;
    float var_64;
    df::float3 var_65;
    float var_66;
    float var_67;
    float var_68;
    float var_69;
    float var_70;
    const float var_71 = 3.0;
    float var_72;
    float var_73;
    float var_74;
    const float var_75 = 0.0;
    bool var_76;
    bool var_77;
    float var_78;
    float var_79;
    mat33 var_80;
    mat33 var_81;
    float var_82;
    mat33 var_83;
    float var_84;
    float var_85;
    float var_86;
    float var_87;
    float var_88;
    df::float3 var_89;
    float var_90;
    float var_91;
    float var_92;
    df::float3 var_93;
    float var_94;
    float var_95;
    float var_96;
    df::float3 var_97;
    df::float3 var_98;
    df::float3 var_99;
    float var_100;
    df::float3 var_101;
    float var_102;
    float var_103;
    float var_104;
    float var_105;
    float var_106;
    float var_107;
    float var_108;
    float var_109;
    float var_110;
    float var_111;
    float var_112;
    float var_113;
    float var_114;
    float var_115;
    float var_116;
    float var_117;
    float var_118;
    df::float3 var_119;
    df::float3 var_120;
    df::float3 var_121;
    df::float3 var_122;
    float var_123;
    float var_124;
    float var_125;
    df::float3 var_126;
    df::float3 var_127;
    df::float3 var_128;
    df::float3 var_129;
    df::float3 var_130;
    df::float3 var_131;
    df::float3 var_132;
    df::float3 var_133;
    float var_134;
    df::float3 var_135;
    float var_136;
    float var_137;
    float var_138;
    float var_139;
    float var_140;
    float var_141;
    float var_142;
    float var_143;
    float var_144;
    float var_145;
    float var_146;
    float var_147;
    float var_148;
    float var_149;
    float var_150;
    float var_151;
    float var_152;
    df::float3 var_153;
    df::float3 var_154;
    df::float3 var_155;
    df::float3 var_156;
    df::float3 var_157;
    df::float3 var_158;
    df::float3 var_159;
    df::float3 var_160;
    df::float3 var_161;
    df::float3 var_162;
    df::float3 var_163;
    df::float3 var_164;
    df::float3 var_165;
    df::float3 var_166;
    df::float3 var_167;
    df::float3 var_168;
    //---------
    // dual vars
    int adj_0 = 0;
    int adj_1 = 0;
    int adj_2 = 0;
    int adj_3 = 0;
    int adj_4 = 0;
    int adj_5 = 0;
    int adj_6 = 0;
    int adj_7 = 0;
    int adj_8 = 0;
    int adj_9 = 0;
    int adj_10 = 0;
    int adj_11 = 0;
    int adj_12 = 0;
    int adj_13 = 0;
    int adj_14 = 0;
    int adj_15 = 0;
    int adj_16 = 0;
    int adj_17 = 0;
    float adj_18 = 0;
    int adj_19 = 0;
    int adj_20 = 0;
    float adj_21 = 0;
    int adj_22 = 0;
    int adj_23 = 0;
    float adj_24 = 0;
    int adj_25 = 0;
    int adj_26 = 0;
    float adj_27 = 0;
    df::float3 adj_28 = 0;
    df::float3 adj_29 = 0;
    df::float3 adj_30 = 0;
    df::float3 adj_31 = 0;
    df::float3 adj_32 = 0;
    df::float3 adj_33 = 0;
    df::float3 adj_34 = 0;
    df::float3 adj_35 = 0;
    float adj_36 = 0;
    float adj_37 = 0;
    float adj_38 = 0;
    float adj_39 = 0;
    df::float3 adj_40 = 0;
    df::float3 adj_41 = 0;
    df::float3 adj_42 = 0;
    df::float3 adj_43 = 0;
    df::float3 adj_44 = 0;
    df::float3 adj_45 = 0;
    mat33 adj_46 = 0;
    mat33 adj_47 = 0;
    float adj_48 = 0;
    float adj_49 = 0;
    float adj_50 = 0;
    float adj_51 = 0;
    float adj_52 = 0;
    mat33 adj_53 = 0;
    float adj_54 = 0;
    float adj_55 = 0;
    float adj_56 = 0;
    df::float3 adj_57 = 0;
    float adj_58 = 0;
    float adj_59 = 0;
    float adj_60 = 0;
    df::float3 adj_61 = 0;
    float adj_62 = 0;
    float adj_63 = 0;
    float adj_64 = 0;
    df::float3 adj_65 = 0;
    float adj_66 = 0;
    float adj_67 = 0;
    float adj_68 = 0;
    float adj_69 = 0;
    float adj_70 = 0;
    float adj_71 = 0;
    float adj_72 = 0;
    float adj_73 = 0;
    float adj_74 = 0;
    float adj_75 = 0;
    bool adj_76 = 0;
    bool adj_77 = 0;
    float adj_78 = 0;
    float adj_79 = 0;
    mat33 adj_80 = 0;
    mat33 adj_81 = 0;
    float adj_82 = 0;
    mat33 adj_83 = 0;
    float adj_84 = 0;
    float adj_85 = 0;
    float adj_86 = 0;
    float adj_87 = 0;
    float adj_88 = 0;
    df::float3 adj_89 = 0;
    float adj_90 = 0;
    float adj_91 = 0;
    float adj_92 = 0;
    df::float3 adj_93 = 0;
    float adj_94 = 0;
    float adj_95 = 0;
    float adj_96 = 0;
    df::float3 adj_97 = 0;
    df::float3 adj_98 = 0;
    df::float3 adj_99 = 0;
    float adj_100 = 0;
    df::float3 adj_101 = 0;
    float adj_102 = 0;
    float adj_103 = 0;
    float adj_104 = 0;
    float adj_105 = 0;
    float adj_106 = 0;
    float adj_107 = 0;
    float adj_108 = 0;
    float adj_109 = 0;
    float adj_110 = 0;
    float adj_111 = 0;
    float adj_112 = 0;
    float adj_113 = 0;
    float adj_114 = 0;
    float adj_115 = 0;
    float adj_116 = 0;
    float adj_117 = 0;
    float adj_118 = 0;
    df::float3 adj_119 = 0;
    df::float3 adj_120 = 0;
    df::float3 adj_121 = 0;
    df::float3 adj_122 = 0;
    float adj_123 = 0;
    float adj_124 = 0;
    float adj_125 = 0;
    df::float3 adj_126 = 0;
    df::float3 adj_127 = 0;
    df::float3 adj_128 = 0;
    df::float3 adj_129 = 0;
    df::float3 adj_130 = 0;
    df::float3 adj_131 = 0;
    df::float3 adj_132 = 0;
    df::float3 adj_133 = 0;
    float adj_134 = 0;
    df::float3 adj_135 = 0;
    float adj_136 = 0;
    float adj_137 = 0;
    float adj_138 = 0;
    float adj_139 = 0;
    float adj_140 = 0;
    float adj_141 = 0;
    float adj_142 = 0;
    float adj_143 = 0;
    float adj_144 = 0;
    float adj_145 = 0;
    float adj_146 = 0;
    float adj_147 = 0;
    float adj_148 = 0;
    float adj_149 = 0;
    float adj_150 = 0;
    float adj_151 = 0;
    float adj_152 = 0;
    df::float3 adj_153 = 0;
    df::float3 adj_154 = 0;
    df::float3 adj_155 = 0;
    df::float3 adj_156 = 0;
    df::float3 adj_157 = 0;
    df::float3 adj_158 = 0;
    df::float3 adj_159 = 0;
    df::float3 adj_160 = 0;
    df::float3 adj_161 = 0;
    df::float3 adj_162 = 0;
    df::float3 adj_163 = 0;
    df::float3 adj_164 = 0;
    df::float3 adj_165 = 0;
    df::float3 adj_166 = 0;
    df::float3 adj_167 = 0;
    df::float3 adj_168 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_2 = df::mul(var_0, var_1);
    var_4 = df::add(var_2, var_3);
    var_5 = df::load(var_indices, var_4);
    var_6 = df::mul(var_0, var_1);
    var_8 = df::add(var_6, var_7);
    var_9 = df::load(var_indices, var_8);
    var_10 = df::mul(var_0, var_1);
    var_12 = df::add(var_10, var_11);
    var_13 = df::load(var_indices, var_12);
    var_14 = df::mul(var_0, var_1);
    var_16 = df::add(var_14, var_15);
    var_17 = df::load(var_indices, var_16);
    var_18 = df::load(var_activation, var_0);
    var_19 = df::mul(var_0, var_15);
    var_20 = df::add(var_19, var_3);
    var_21 = df::load(var_materials, var_20);
    var_22 = df::mul(var_0, var_15);
    var_23 = df::add(var_22, var_7);
    var_24 = df::load(var_materials, var_23);
    var_25 = df::mul(var_0, var_15);
    var_26 = df::add(var_25, var_11);
    var_27 = df::load(var_materials, var_26);
    var_28 = df::load(var_x, var_5);
    var_29 = df::load(var_x, var_9);
    var_30 = df::load(var_x, var_13);
    var_31 = df::load(var_x, var_17);
    var_32 = df::load(var_v, var_5);
    var_33 = df::load(var_v, var_9);
    var_34 = df::load(var_v, var_13);
    var_35 = df::load(var_v, var_17);
    var_36 = df::load(var_inv_mass, var_5);
    var_37 = df::load(var_inv_mass, var_9);
    var_38 = df::load(var_inv_mass, var_13);
    var_39 = df::load(var_inv_mass, var_17);
    var_40 = df::sub(var_29, var_28);
    var_41 = df::sub(var_30, var_28);
    var_42 = df::sub(var_31, var_28);
    var_43 = df::sub(var_33, var_32);
    var_44 = df::sub(var_34, var_32);
    var_45 = df::sub(var_35, var_32);
    var_46 = df::mat33(var_40, var_41, var_42);
    var_47 = df::load(var_pose, var_0);
    var_48 = df::determinant(var_47);
    var_50 = df::mul(var_48, var_49);
    var_52 = df::div(var_51, var_50);
    var_53 = df::mul(var_46, var_47);
    var_54 = df::index(var_53, var_3, var_3);
    var_55 = df::index(var_53, var_7, var_3);
    var_56 = df::index(var_53, var_11, var_3);
    var_57 = df::float3(var_54, var_55, var_56);
    var_58 = df::index(var_53, var_3, var_7);
    var_59 = df::index(var_53, var_7, var_7);
    var_60 = df::index(var_53, var_11, var_7);
    var_61 = df::float3(var_58, var_59, var_60);
    var_62 = df::index(var_53, var_3, var_11);
    var_63 = df::index(var_53, var_7, var_11);
    var_64 = df::index(var_53, var_11, var_11);
    var_65 = df::float3(var_62, var_63, var_64);
    var_66 = df::dot(var_57, var_57);
    var_67 = df::dot(var_61, var_61);
    var_68 = df::add(var_66, var_67);
    var_69 = df::dot(var_65, var_65);
    var_70 = df::add(var_68, var_69);
    var_72 = df::sub(var_70, var_71);
    var_73 = df::abs(var_72);
    var_74 = df::sqrt(var_73);
    var_76 = (var_74 == var_75);
    if (var_76) {
    	goto label0;
    }
    var_77 = (var_70 < var_71);
    if (var_77) {
    	var_78 = df::sub(var_75, var_74);
    }
    var_79 = df::select(var_77, var_74, var_78);
    var_80 = df::transpose(var_47);
    var_81 = df::mul(var_53, var_80);
    var_82 = df::div(var_51, var_79);
    var_83 = df::mul(var_81, var_82);
    var_84 = df::div(var_21, var_24);
    var_85 = df::add(var_51, var_84);
    var_86 = df::index(var_83, var_3, var_3);
    var_87 = df::index(var_83, var_7, var_3);
    var_88 = df::index(var_83, var_11, var_3);
    var_89 = df::float3(var_86, var_87, var_88);
    var_90 = df::index(var_83, var_3, var_7);
    var_91 = df::index(var_83, var_7, var_7);
    var_92 = df::index(var_83, var_11, var_7);
    var_93 = df::float3(var_90, var_91, var_92);
    var_94 = df::index(var_83, var_3, var_11);
    var_95 = df::index(var_83, var_7, var_11);
    var_96 = df::index(var_83, var_11, var_11);
    var_97 = df::float3(var_94, var_95, var_96);
    var_98 = df::add(var_89, var_93);
    var_99 = df::add(var_98, var_97);
    var_100 = df::sub(var_75, var_51);
    var_101 = df::mul(var_99, var_100);
    var_102 = df::dot(var_101, var_101);
    var_103 = df::mul(var_102, var_36);
    var_104 = df::dot(var_89, var_89);
    var_105 = df::mul(var_104, var_37);
    var_106 = df::add(var_103, var_105);
    var_107 = df::dot(var_93, var_93);
    var_108 = df::mul(var_107, var_38);
    var_109 = df::add(var_106, var_108);
    var_110 = df::dot(var_97, var_97);
    var_111 = df::mul(var_110, var_39);
    var_112 = df::add(var_109, var_111);
    var_113 = df::mul(var_21, var_dt);
    var_114 = df::mul(var_113, var_dt);
    var_115 = df::mul(var_114, var_52);
    var_116 = df::div(var_51, var_115);
    var_117 = df::add(var_112, var_116);
    var_118 = df::div(var_74, var_117);
    var_119 = df::mul(var_101, var_118);
    var_120 = df::mul(var_89, var_118);
    var_121 = df::mul(var_93, var_118);
    var_122 = df::mul(var_97, var_118);
    var_123 = df::determinant(var_53);
    var_124 = df::sub(var_123, var_85);
    var_125 = df::div(var_50, var_49);
    var_126 = df::cross(var_41, var_42);
    var_127 = df::mul(var_126, var_125);
    var_128 = df::cross(var_42, var_40);
    var_129 = df::mul(var_128, var_125);
    var_130 = df::cross(var_40, var_41);
    var_131 = df::mul(var_130, var_125);
    var_132 = df::add(var_127, var_129);
    var_133 = df::add(var_132, var_131);
    var_134 = df::sub(var_75, var_51);
    var_135 = df::mul(var_133, var_134);
    var_136 = df::dot(var_135, var_135);
    var_137 = df::mul(var_136, var_36);
    var_138 = df::dot(var_127, var_127);
    var_139 = df::mul(var_138, var_37);
    var_140 = df::add(var_137, var_139);
    var_141 = df::dot(var_129, var_129);
    var_142 = df::mul(var_141, var_38);
    var_143 = df::add(var_140, var_142);
    var_144 = df::dot(var_131, var_131);
    var_145 = df::mul(var_144, var_39);
    var_146 = df::add(var_143, var_145);
    var_147 = df::mul(var_24, var_dt);
    var_148 = df::mul(var_147, var_dt);
    var_149 = df::mul(var_148, var_52);
    var_150 = df::div(var_51, var_149);
    var_151 = df::add(var_146, var_150);
    var_152 = df::div(var_124, var_151);
    var_153 = df::mul(var_135, var_152);
    var_154 = df::add(var_119, var_153);
    var_155 = df::mul(var_127, var_152);
    var_156 = df::add(var_120, var_155);
    var_157 = df::mul(var_129, var_152);
    var_158 = df::add(var_121, var_157);
    var_159 = df::mul(var_131, var_152);
    var_160 = df::add(var_122, var_159);
    var_161 = df::mul(var_154, var_36);
    var_162 = df::mul(var_161, var_relaxation);
    df::atomic_sub(var_delta, var_5, var_162);
    var_163 = df::mul(var_156, var_37);
    var_164 = df::mul(var_163, var_relaxation);
    df::atomic_sub(var_delta, var_9, var_164);
    var_165 = df::mul(var_158, var_38);
    var_166 = df::mul(var_165, var_relaxation);
    df::atomic_sub(var_delta, var_13, var_166);
    var_167 = df::mul(var_160, var_39);
    var_168 = df::mul(var_167, var_relaxation);
    df::atomic_sub(var_delta, var_17, var_168);
    //---------
    // reverse
    df::adj_atomic_sub(var_delta, var_17, var_168, adj_delta, adj_17, adj_168);
    df::adj_mul(var_167, var_relaxation, adj_167, adj_relaxation, adj_168);
    df::adj_mul(var_160, var_39, adj_160, adj_39, adj_167);
    df::adj_atomic_sub(var_delta, var_13, var_166, adj_delta, adj_13, adj_166);
    df::adj_mul(var_165, var_relaxation, adj_165, adj_relaxation, adj_166);
    df::adj_mul(var_158, var_38, adj_158, adj_38, adj_165);
    df::adj_atomic_sub(var_delta, var_9, var_164, adj_delta, adj_9, adj_164);
    df::adj_mul(var_163, var_relaxation, adj_163, adj_relaxation, adj_164);
    df::adj_mul(var_156, var_37, adj_156, adj_37, adj_163);
    df::adj_atomic_sub(var_delta, var_5, var_162, adj_delta, adj_5, adj_162);
    df::adj_mul(var_161, var_relaxation, adj_161, adj_relaxation, adj_162);
    df::adj_mul(var_154, var_36, adj_154, adj_36, adj_161);
    df::adj_add(var_122, var_159, adj_122, adj_159, adj_160);
    df::adj_mul(var_131, var_152, adj_131, adj_152, adj_159);
    df::adj_add(var_121, var_157, adj_121, adj_157, adj_158);
    df::adj_mul(var_129, var_152, adj_129, adj_152, adj_157);
    df::adj_add(var_120, var_155, adj_120, adj_155, adj_156);
    df::adj_mul(var_127, var_152, adj_127, adj_152, adj_155);
    df::adj_add(var_119, var_153, adj_119, adj_153, adj_154);
    df::adj_mul(var_135, var_152, adj_135, adj_152, adj_153);
    df::adj_div(var_124, var_151, adj_124, adj_151, adj_152);
    df::adj_add(var_146, var_150, adj_146, adj_150, adj_151);
    df::adj_div(var_51, var_149, adj_51, adj_149, adj_150);
    df::adj_mul(var_148, var_52, adj_148, adj_52, adj_149);
    df::adj_mul(var_147, var_dt, adj_147, adj_dt, adj_148);
    df::adj_mul(var_24, var_dt, adj_24, adj_dt, adj_147);
    df::adj_add(var_143, var_145, adj_143, adj_145, adj_146);
    df::adj_mul(var_144, var_39, adj_144, adj_39, adj_145);
    df::adj_dot(var_131, var_131, adj_131, adj_131, adj_144);
    df::adj_add(var_140, var_142, adj_140, adj_142, adj_143);
    df::adj_mul(var_141, var_38, adj_141, adj_38, adj_142);
    df::adj_dot(var_129, var_129, adj_129, adj_129, adj_141);
    df::adj_add(var_137, var_139, adj_137, adj_139, adj_140);
    df::adj_mul(var_138, var_37, adj_138, adj_37, adj_139);
    df::adj_dot(var_127, var_127, adj_127, adj_127, adj_138);
    df::adj_mul(var_136, var_36, adj_136, adj_36, adj_137);
    df::adj_dot(var_135, var_135, adj_135, adj_135, adj_136);
    df::adj_mul(var_133, var_134, adj_133, adj_134, adj_135);
    df::adj_sub(var_75, var_51, adj_75, adj_51, adj_134);
    df::adj_add(var_132, var_131, adj_132, adj_131, adj_133);
    df::adj_add(var_127, var_129, adj_127, adj_129, adj_132);
    df::adj_mul(var_130, var_125, adj_130, adj_125, adj_131);
    df::adj_cross(var_40, var_41, adj_40, adj_41, adj_130);
    df::adj_mul(var_128, var_125, adj_128, adj_125, adj_129);
    df::adj_cross(var_42, var_40, adj_42, adj_40, adj_128);
    df::adj_mul(var_126, var_125, adj_126, adj_125, adj_127);
    df::adj_cross(var_41, var_42, adj_41, adj_42, adj_126);
    df::adj_div(var_50, var_49, adj_50, adj_49, adj_125);
    df::adj_sub(var_123, var_85, adj_123, adj_85, adj_124);
    df::adj_determinant(var_53, adj_53, adj_123);
    df::adj_mul(var_97, var_118, adj_97, adj_118, adj_122);
    df::adj_mul(var_93, var_118, adj_93, adj_118, adj_121);
    df::adj_mul(var_89, var_118, adj_89, adj_118, adj_120);
    df::adj_mul(var_101, var_118, adj_101, adj_118, adj_119);
    df::adj_div(var_74, var_117, adj_74, adj_117, adj_118);
    df::adj_add(var_112, var_116, adj_112, adj_116, adj_117);
    df::adj_div(var_51, var_115, adj_51, adj_115, adj_116);
    df::adj_mul(var_114, var_52, adj_114, adj_52, adj_115);
    df::adj_mul(var_113, var_dt, adj_113, adj_dt, adj_114);
    df::adj_mul(var_21, var_dt, adj_21, adj_dt, adj_113);
    df::adj_add(var_109, var_111, adj_109, adj_111, adj_112);
    df::adj_mul(var_110, var_39, adj_110, adj_39, adj_111);
    df::adj_dot(var_97, var_97, adj_97, adj_97, adj_110);
    df::adj_add(var_106, var_108, adj_106, adj_108, adj_109);
    df::adj_mul(var_107, var_38, adj_107, adj_38, adj_108);
    df::adj_dot(var_93, var_93, adj_93, adj_93, adj_107);
    df::adj_add(var_103, var_105, adj_103, adj_105, adj_106);
    df::adj_mul(var_104, var_37, adj_104, adj_37, adj_105);
    df::adj_dot(var_89, var_89, adj_89, adj_89, adj_104);
    df::adj_mul(var_102, var_36, adj_102, adj_36, adj_103);
    df::adj_dot(var_101, var_101, adj_101, adj_101, adj_102);
    df::adj_mul(var_99, var_100, adj_99, adj_100, adj_101);
    df::adj_sub(var_75, var_51, adj_75, adj_51, adj_100);
    df::adj_add(var_98, var_97, adj_98, adj_97, adj_99);
    df::adj_add(var_89, var_93, adj_89, adj_93, adj_98);
    df::adj_float3(var_94, var_95, var_96, adj_94, adj_95, adj_96, adj_97);
    df::adj_index(var_83, var_11, var_11, adj_83, adj_11, adj_11, adj_96);
    df::adj_index(var_83, var_7, var_11, adj_83, adj_7, adj_11, adj_95);
    df::adj_index(var_83, var_3, var_11, adj_83, adj_3, adj_11, adj_94);
    df::adj_float3(var_90, var_91, var_92, adj_90, adj_91, adj_92, adj_93);
    df::adj_index(var_83, var_11, var_7, adj_83, adj_11, adj_7, adj_92);
    df::adj_index(var_83, var_7, var_7, adj_83, adj_7, adj_7, adj_91);
    df::adj_index(var_83, var_3, var_7, adj_83, adj_3, adj_7, adj_90);
    df::adj_float3(var_86, var_87, var_88, adj_86, adj_87, adj_88, adj_89);
    df::adj_index(var_83, var_11, var_3, adj_83, adj_11, adj_3, adj_88);
    df::adj_index(var_83, var_7, var_3, adj_83, adj_7, adj_3, adj_87);
    df::adj_index(var_83, var_3, var_3, adj_83, adj_3, adj_3, adj_86);
    df::adj_add(var_51, var_84, adj_51, adj_84, adj_85);
    df::adj_div(var_21, var_24, adj_21, adj_24, adj_84);
    df::adj_mul(var_81, var_82, adj_81, adj_82, adj_83);
    df::adj_div(var_51, var_79, adj_51, adj_79, adj_82);
    df::adj_mul(var_53, var_80, adj_53, adj_80, adj_81);
    df::adj_transpose(var_47, adj_47, adj_80);
    df::adj_select(var_77, var_74, var_78, adj_77, adj_74, adj_78, adj_79);
    if (var_77) {
    	df::adj_sub(var_75, var_74, adj_75, adj_74, adj_78);
    }
    if (var_76) {
    	label0:;
    }
    df::adj_sqrt(var_73, adj_73, adj_74);
    df::adj_abs(var_72, adj_72, adj_73);
    df::adj_sub(var_70, var_71, adj_70, adj_71, adj_72);
    df::adj_add(var_68, var_69, adj_68, adj_69, adj_70);
    df::adj_dot(var_65, var_65, adj_65, adj_65, adj_69);
    df::adj_add(var_66, var_67, adj_66, adj_67, adj_68);
    df::adj_dot(var_61, var_61, adj_61, adj_61, adj_67);
    df::adj_dot(var_57, var_57, adj_57, adj_57, adj_66);
    df::adj_float3(var_62, var_63, var_64, adj_62, adj_63, adj_64, adj_65);
    df::adj_index(var_53, var_11, var_11, adj_53, adj_11, adj_11, adj_64);
    df::adj_index(var_53, var_7, var_11, adj_53, adj_7, adj_11, adj_63);
    df::adj_index(var_53, var_3, var_11, adj_53, adj_3, adj_11, adj_62);
    df::adj_float3(var_58, var_59, var_60, adj_58, adj_59, adj_60, adj_61);
    df::adj_index(var_53, var_11, var_7, adj_53, adj_11, adj_7, adj_60);
    df::adj_index(var_53, var_7, var_7, adj_53, adj_7, adj_7, adj_59);
    df::adj_index(var_53, var_3, var_7, adj_53, adj_3, adj_7, adj_58);
    df::adj_float3(var_54, var_55, var_56, adj_54, adj_55, adj_56, adj_57);
    df::adj_index(var_53, var_11, var_3, adj_53, adj_11, adj_3, adj_56);
    df::adj_index(var_53, var_7, var_3, adj_53, adj_7, adj_3, adj_55);
    df::adj_index(var_53, var_3, var_3, adj_53, adj_3, adj_3, adj_54);
    df::adj_mul(var_46, var_47, adj_46, adj_47, adj_53);
    df::adj_div(var_51, var_50, adj_51, adj_50, adj_52);
    df::adj_mul(var_48, var_49, adj_48, adj_49, adj_50);
    df::adj_determinant(var_47, adj_47, adj_48);
    df::adj_load(var_pose, var_0, adj_pose, adj_0, adj_47);
    df::adj_mat33(var_40, var_41, var_42, adj_40, adj_41, adj_42, adj_46);
    df::adj_sub(var_35, var_32, adj_35, adj_32, adj_45);
    df::adj_sub(var_34, var_32, adj_34, adj_32, adj_44);
    df::adj_sub(var_33, var_32, adj_33, adj_32, adj_43);
    df::adj_sub(var_31, var_28, adj_31, adj_28, adj_42);
    df::adj_sub(var_30, var_28, adj_30, adj_28, adj_41);
    df::adj_sub(var_29, var_28, adj_29, adj_28, adj_40);
    df::adj_load(var_inv_mass, var_17, adj_inv_mass, adj_17, adj_39);
    df::adj_load(var_inv_mass, var_13, adj_inv_mass, adj_13, adj_38);
    df::adj_load(var_inv_mass, var_9, adj_inv_mass, adj_9, adj_37);
    df::adj_load(var_inv_mass, var_5, adj_inv_mass, adj_5, adj_36);
    df::adj_load(var_v, var_17, adj_v, adj_17, adj_35);
    df::adj_load(var_v, var_13, adj_v, adj_13, adj_34);
    df::adj_load(var_v, var_9, adj_v, adj_9, adj_33);
    df::adj_load(var_v, var_5, adj_v, adj_5, adj_32);
    df::adj_load(var_x, var_17, adj_x, adj_17, adj_31);
    df::adj_load(var_x, var_13, adj_x, adj_13, adj_30);
    df::adj_load(var_x, var_9, adj_x, adj_9, adj_29);
    df::adj_load(var_x, var_5, adj_x, adj_5, adj_28);
    df::adj_load(var_materials, var_26, adj_materials, adj_26, adj_27);
    df::adj_add(var_25, var_11, adj_25, adj_11, adj_26);
    df::adj_mul(var_0, var_15, adj_0, adj_15, adj_25);
    df::adj_load(var_materials, var_23, adj_materials, adj_23, adj_24);
    df::adj_add(var_22, var_7, adj_22, adj_7, adj_23);
    df::adj_mul(var_0, var_15, adj_0, adj_15, adj_22);
    df::adj_load(var_materials, var_20, adj_materials, adj_20, adj_21);
    df::adj_add(var_19, var_3, adj_19, adj_3, adj_20);
    df::adj_mul(var_0, var_15, adj_0, adj_15, adj_19);
    df::adj_load(var_activation, var_0, adj_activation, adj_0, adj_18);
    df::adj_load(var_indices, var_16, adj_indices, adj_16, adj_17);
    df::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_14);
    df::adj_load(var_indices, var_12, adj_indices, adj_12, adj_13);
    df::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_10);
    df::adj_load(var_indices, var_8, adj_indices, adj_8, adj_9);
    df::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_6);
    df::adj_load(var_indices, var_4, adj_indices, adj_4, adj_5);
    df::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    return;

}



// Python entry points
void solve_tetrahedra_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	float var_dt,
	float var_relaxation,
	torch::Tensor var_delta)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        solve_tetrahedra_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<float*>(var_inv_mass),
			cast<int*>(var_indices),
			cast<mat33*>(var_pose),
			cast<float*>(var_activation),
			cast<float*>(var_materials),
			var_dt,
			var_relaxation,
			cast<df::float3*>(var_delta));
    }
}

void solve_tetrahedra_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	float var_dt,
	float var_relaxation,
	torch::Tensor var_delta, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_inv_mass,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	torch::Tensor adj_materials,
	float adj_dt,
	float adj_relaxation,
	torch::Tensor adj_delta)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        solve_tetrahedra_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<float*>(var_inv_mass),
			cast<int*>(var_indices),
			cast<mat33*>(var_pose),
			cast<float*>(var_activation),
			cast<float*>(var_materials),
			var_dt,
			var_relaxation,
			cast<df::float3*>(var_delta), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<float*>(adj_inv_mass),
			cast<int*>(adj_indices),
			cast<mat33*>(adj_pose),
			cast<float*>(adj_activation),
			cast<float*>(adj_materials),
			adj_dt,
			adj_relaxation,
			cast<df::float3*>(adj_delta));
    }
}



// Python entry points
void solve_tetrahedra_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	float var_dt,
	float var_relaxation,
	torch::Tensor var_delta);

void solve_tetrahedra_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	torch::Tensor var_indices,
	torch::Tensor var_pose,
	torch::Tensor var_activation,
	torch::Tensor var_materials,
	float var_dt,
	float var_relaxation,
	torch::Tensor var_delta, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_inv_mass,
	torch::Tensor adj_indices,
	torch::Tensor adj_pose,
	torch::Tensor adj_activation,
	torch::Tensor adj_materials,
	float adj_dt,
	float adj_relaxation,
	torch::Tensor adj_delta);


void solve_contacts_cpu_kernel_forward(
	df::float3* var_x,
	df::float3* var_v,
	float* var_inv_mass,
	float var_mu,
	float var_dt,
	df::float3* var_delta)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    float var_3;
    const float var_4 = 0.0;
    const float var_5 = 1.0;
    df::float3 var_6;
    float var_7;
    const float var_8 = 0.01;
    float var_9;
    bool var_10;
    df::float3 var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    float var_15;
    float var_16;
    float var_17;
    float var_18;
    float var_19;
    df::float3 var_20;
    df::float3 var_21;
    df::float3 var_22;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x, var_0);
    var_2 = df::load(var_v, var_0);
    var_3 = df::load(var_inv_mass, var_0);
    var_6 = df::float3(var_4, var_5, var_4);
    var_7 = df::dot(var_6, var_1);
    var_9 = df::sub(var_7, var_8);
    var_10 = (var_9 > var_4);
    if (var_10) {
    	return;
    }
    var_11 = df::mul(var_6, var_9);
    var_12 = df::dot(var_6, var_2);
    var_13 = df::mul(var_6, var_12);
    var_14 = df::sub(var_2, var_13);
    var_15 = df::mul(var_mu, var_9);
    var_16 = df::length(var_14);
    var_17 = df::mul(var_16, var_dt);
    var_18 = df::sub(var_4, var_17);
    var_19 = df::max(var_15, var_18);
    var_20 = df::normalize(var_14);
    var_21 = df::mul(var_20, var_19);
    var_22 = df::sub(var_21, var_11);
    df::atomic_add(var_delta, var_0, var_22);
    

}

void solve_contacts_cpu_kernel_backward(
	df::float3* var_x,
	df::float3* var_v,
	float* var_inv_mass,
	float var_mu,
	float var_dt,
	df::float3* var_delta, 
	df::float3* adj_x,
	df::float3* adj_v,
	float* adj_inv_mass,
	float adj_mu,
	float adj_dt,
	df::float3* adj_delta)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    float var_3;
    const float var_4 = 0.0;
    const float var_5 = 1.0;
    df::float3 var_6;
    float var_7;
    const float var_8 = 0.01;
    float var_9;
    bool var_10;
    df::float3 var_11;
    float var_12;
    df::float3 var_13;
    df::float3 var_14;
    float var_15;
    float var_16;
    float var_17;
    float var_18;
    float var_19;
    df::float3 var_20;
    df::float3 var_21;
    df::float3 var_22;
    //---------
    // dual vars
    int adj_0 = 0;
    df::float3 adj_1 = 0;
    df::float3 adj_2 = 0;
    float adj_3 = 0;
    float adj_4 = 0;
    float adj_5 = 0;
    df::float3 adj_6 = 0;
    float adj_7 = 0;
    float adj_8 = 0;
    float adj_9 = 0;
    bool adj_10 = 0;
    df::float3 adj_11 = 0;
    float adj_12 = 0;
    df::float3 adj_13 = 0;
    df::float3 adj_14 = 0;
    float adj_15 = 0;
    float adj_16 = 0;
    float adj_17 = 0;
    float adj_18 = 0;
    float adj_19 = 0;
    df::float3 adj_20 = 0;
    df::float3 adj_21 = 0;
    df::float3 adj_22 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x, var_0);
    var_2 = df::load(var_v, var_0);
    var_3 = df::load(var_inv_mass, var_0);
    var_6 = df::float3(var_4, var_5, var_4);
    var_7 = df::dot(var_6, var_1);
    var_9 = df::sub(var_7, var_8);
    var_10 = (var_9 > var_4);
    if (var_10) {
    	goto label0;
    }
    var_11 = df::mul(var_6, var_9);
    var_12 = df::dot(var_6, var_2);
    var_13 = df::mul(var_6, var_12);
    var_14 = df::sub(var_2, var_13);
    var_15 = df::mul(var_mu, var_9);
    var_16 = df::length(var_14);
    var_17 = df::mul(var_16, var_dt);
    var_18 = df::sub(var_4, var_17);
    var_19 = df::max(var_15, var_18);
    var_20 = df::normalize(var_14);
    var_21 = df::mul(var_20, var_19);
    var_22 = df::sub(var_21, var_11);
    df::atomic_add(var_delta, var_0, var_22);
    //---------
    // reverse
    df::adj_atomic_add(var_delta, var_0, var_22, adj_delta, adj_0, adj_22);
    df::adj_sub(var_21, var_11, adj_21, adj_11, adj_22);
    df::adj_mul(var_20, var_19, adj_20, adj_19, adj_21);
    df::adj_normalize(var_14, adj_14, adj_20);
    df::adj_max(var_15, var_18, adj_15, adj_18, adj_19);
    df::adj_sub(var_4, var_17, adj_4, adj_17, adj_18);
    df::adj_mul(var_16, var_dt, adj_16, adj_dt, adj_17);
    df::adj_length(var_14, adj_14, adj_16);
    df::adj_mul(var_mu, var_9, adj_mu, adj_9, adj_15);
    df::adj_sub(var_2, var_13, adj_2, adj_13, adj_14);
    df::adj_mul(var_6, var_12, adj_6, adj_12, adj_13);
    df::adj_dot(var_6, var_2, adj_6, adj_2, adj_12);
    df::adj_mul(var_6, var_9, adj_6, adj_9, adj_11);
    if (var_10) {
    	label0:;
    }
    df::adj_sub(var_7, var_8, adj_7, adj_8, adj_9);
    df::adj_dot(var_6, var_1, adj_6, adj_1, adj_7);
    df::adj_float3(var_4, var_5, var_4, adj_4, adj_5, adj_4, adj_6);
    df::adj_load(var_inv_mass, var_0, adj_inv_mass, adj_0, adj_3);
    df::adj_load(var_v, var_0, adj_v, adj_0, adj_2);
    df::adj_load(var_x, var_0, adj_x, adj_0, adj_1);
    return;

}



// Python entry points
void solve_contacts_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	float var_mu,
	float var_dt,
	torch::Tensor var_delta)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        solve_contacts_cpu_kernel_forward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<float*>(var_inv_mass),
			var_mu,
			var_dt,
			cast<df::float3*>(var_delta));
    }
}

void solve_contacts_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	float var_mu,
	float var_dt,
	torch::Tensor var_delta, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_inv_mass,
	float adj_mu,
	float adj_dt,
	torch::Tensor adj_delta)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        solve_contacts_cpu_kernel_backward(
			cast<df::float3*>(var_x),
			cast<df::float3*>(var_v),
			cast<float*>(var_inv_mass),
			var_mu,
			var_dt,
			cast<df::float3*>(var_delta), 
			cast<df::float3*>(adj_x),
			cast<df::float3*>(adj_v),
			cast<float*>(adj_inv_mass),
			adj_mu,
			adj_dt,
			cast<df::float3*>(adj_delta));
    }
}



// Python entry points
void solve_contacts_cpu_forward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	float var_mu,
	float var_dt,
	torch::Tensor var_delta);

void solve_contacts_cpu_backward(int dim, 
	torch::Tensor var_x,
	torch::Tensor var_v,
	torch::Tensor var_inv_mass,
	float var_mu,
	float var_dt,
	torch::Tensor var_delta, 
	torch::Tensor adj_x,
	torch::Tensor adj_v,
	torch::Tensor adj_inv_mass,
	float adj_mu,
	float adj_dt,
	torch::Tensor adj_delta);


void apply_deltas_cpu_kernel_forward(
	df::float3* var_x_orig,
	df::float3* var_v_orig,
	df::float3* var_x_pred,
	df::float3* var_delta,
	float var_dt,
	df::float3* var_x_out,
	df::float3* var_v_out)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x_orig, var_0);
    var_2 = df::load(var_x_pred, var_0);
    var_3 = df::load(var_delta, var_0);
    var_4 = df::add(var_2, var_3);
    var_5 = df::sub(var_4, var_1);
    var_6 = df::div(var_5, var_dt);
    df::store(var_x_out, var_0, var_4);
    df::store(var_v_out, var_0, var_6);
    

}

void apply_deltas_cpu_kernel_backward(
	df::float3* var_x_orig,
	df::float3* var_v_orig,
	df::float3* var_x_pred,
	df::float3* var_delta,
	float var_dt,
	df::float3* var_x_out,
	df::float3* var_v_out, 
	df::float3* adj_x_orig,
	df::float3* adj_v_orig,
	df::float3* adj_x_pred,
	df::float3* adj_delta,
	float adj_dt,
	df::float3* adj_x_out,
	df::float3* adj_v_out)
{
        //---------
    // primal vars
    int var_0;
    df::float3 var_1;
    df::float3 var_2;
    df::float3 var_3;
    df::float3 var_4;
    df::float3 var_5;
    df::float3 var_6;
    //---------
    // dual vars
    int adj_0 = 0;
    df::float3 adj_1 = 0;
    df::float3 adj_2 = 0;
    df::float3 adj_3 = 0;
    df::float3 adj_4 = 0;
    df::float3 adj_5 = 0;
    df::float3 adj_6 = 0;
    //---------
    // forward
    var_0 = df::tid();
    var_1 = df::load(var_x_orig, var_0);
    var_2 = df::load(var_x_pred, var_0);
    var_3 = df::load(var_delta, var_0);
    var_4 = df::add(var_2, var_3);
    var_5 = df::sub(var_4, var_1);
    var_6 = df::div(var_5, var_dt);
    df::store(var_x_out, var_0, var_4);
    df::store(var_v_out, var_0, var_6);
    //---------
    // reverse
    df::adj_store(var_v_out, var_0, var_6, adj_v_out, adj_0, adj_6);
    df::adj_store(var_x_out, var_0, var_4, adj_x_out, adj_0, adj_4);
    df::adj_div(var_5, var_dt, adj_5, adj_dt, adj_6);
    df::adj_sub(var_4, var_1, adj_4, adj_1, adj_5);
    df::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    df::adj_load(var_delta, var_0, adj_delta, adj_0, adj_3);
    df::adj_load(var_x_pred, var_0, adj_x_pred, adj_0, adj_2);
    df::adj_load(var_x_orig, var_0, adj_x_orig, adj_0, adj_1);
    return;

}



// Python entry points
void apply_deltas_cpu_forward(int dim, 
	torch::Tensor var_x_orig,
	torch::Tensor var_v_orig,
	torch::Tensor var_x_pred,
	torch::Tensor var_delta,
	float var_dt,
	torch::Tensor var_x_out,
	torch::Tensor var_v_out)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        apply_deltas_cpu_kernel_forward(
			cast<df::float3*>(var_x_orig),
			cast<df::float3*>(var_v_orig),
			cast<df::float3*>(var_x_pred),
			cast<df::float3*>(var_delta),
			var_dt,
			cast<df::float3*>(var_x_out),
			cast<df::float3*>(var_v_out));
    }
}

void apply_deltas_cpu_backward(int dim, 
	torch::Tensor var_x_orig,
	torch::Tensor var_v_orig,
	torch::Tensor var_x_pred,
	torch::Tensor var_delta,
	float var_dt,
	torch::Tensor var_x_out,
	torch::Tensor var_v_out, 
	torch::Tensor adj_x_orig,
	torch::Tensor adj_v_orig,
	torch::Tensor adj_x_pred,
	torch::Tensor adj_delta,
	float adj_dt,
	torch::Tensor adj_x_out,
	torch::Tensor adj_v_out)
{
    for (int i=0; i < dim; ++i)
    {
        s_threadIdx = i;

        apply_deltas_cpu_kernel_backward(
			cast<df::float3*>(var_x_orig),
			cast<df::float3*>(var_v_orig),
			cast<df::float3*>(var_x_pred),
			cast<df::float3*>(var_delta),
			var_dt,
			cast<df::float3*>(var_x_out),
			cast<df::float3*>(var_v_out), 
			cast<df::float3*>(adj_x_orig),
			cast<df::float3*>(adj_v_orig),
			cast<df::float3*>(adj_x_pred),
			cast<df::float3*>(adj_delta),
			adj_dt,
			cast<df::float3*>(adj_x_out),
			cast<df::float3*>(adj_v_out));
    }
}



// Python entry points
void apply_deltas_cpu_forward(int dim, 
	torch::Tensor var_x_orig,
	torch::Tensor var_v_orig,
	torch::Tensor var_x_pred,
	torch::Tensor var_delta,
	float var_dt,
	torch::Tensor var_x_out,
	torch::Tensor var_v_out);

void apply_deltas_cpu_backward(int dim, 
	torch::Tensor var_x_orig,
	torch::Tensor var_v_orig,
	torch::Tensor var_x_pred,
	torch::Tensor var_delta,
	float var_dt,
	torch::Tensor var_x_out,
	torch::Tensor var_v_out, 
	torch::Tensor adj_x_orig,
	torch::Tensor adj_v_orig,
	torch::Tensor adj_x_pred,
	torch::Tensor adj_delta,
	float adj_dt,
	torch::Tensor adj_x_out,
	torch::Tensor adj_v_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("integrate_particles_cpu_forward", integrate_particles_cpu_forward, "integrate_particles_cpu_forward");
m.def("integrate_particles_cpu_backward", integrate_particles_cpu_backward, "integrate_particles_cpu_backward");
m.def("integrate_rigids_cpu_forward", integrate_rigids_cpu_forward, "integrate_rigids_cpu_forward");
m.def("integrate_rigids_cpu_backward", integrate_rigids_cpu_backward, "integrate_rigids_cpu_backward");
m.def("eval_springs_cpu_forward", eval_springs_cpu_forward, "eval_springs_cpu_forward");
m.def("eval_springs_cpu_backward", eval_springs_cpu_backward, "eval_springs_cpu_backward");
m.def("eval_triangles_cpu_forward", eval_triangles_cpu_forward, "eval_triangles_cpu_forward");
m.def("eval_triangles_cpu_backward", eval_triangles_cpu_backward, "eval_triangles_cpu_backward");
m.def("eval_triangles_contact_cpu_forward", eval_triangles_contact_cpu_forward, "eval_triangles_contact_cpu_forward");
m.def("eval_triangles_contact_cpu_backward", eval_triangles_contact_cpu_backward, "eval_triangles_contact_cpu_backward");
m.def("eval_triangles_rigid_contacts_cpu_forward", eval_triangles_rigid_contacts_cpu_forward, "eval_triangles_rigid_contacts_cpu_forward");
m.def("eval_triangles_rigid_contacts_cpu_backward", eval_triangles_rigid_contacts_cpu_backward, "eval_triangles_rigid_contacts_cpu_backward");
m.def("eval_bending_cpu_forward", eval_bending_cpu_forward, "eval_bending_cpu_forward");
m.def("eval_bending_cpu_backward", eval_bending_cpu_backward, "eval_bending_cpu_backward");
m.def("eval_tetrahedra_cpu_forward", eval_tetrahedra_cpu_forward, "eval_tetrahedra_cpu_forward");
m.def("eval_tetrahedra_cpu_backward", eval_tetrahedra_cpu_backward, "eval_tetrahedra_cpu_backward");
m.def("eval_contacts_cpu_forward", eval_contacts_cpu_forward, "eval_contacts_cpu_forward");
m.def("eval_contacts_cpu_backward", eval_contacts_cpu_backward, "eval_contacts_cpu_backward");
m.def("eval_soft_contacts_cpu_forward", eval_soft_contacts_cpu_forward, "eval_soft_contacts_cpu_forward");
m.def("eval_soft_contacts_cpu_backward", eval_soft_contacts_cpu_backward, "eval_soft_contacts_cpu_backward");
m.def("eval_rigid_contacts_cpu_forward", eval_rigid_contacts_cpu_forward, "eval_rigid_contacts_cpu_forward");
m.def("eval_rigid_contacts_cpu_backward", eval_rigid_contacts_cpu_backward, "eval_rigid_contacts_cpu_backward");
m.def("eval_rigid_contacts_art_cpu_forward", eval_rigid_contacts_art_cpu_forward, "eval_rigid_contacts_art_cpu_forward");
m.def("eval_rigid_contacts_art_cpu_backward", eval_rigid_contacts_art_cpu_backward, "eval_rigid_contacts_art_cpu_backward");
m.def("eval_muscles_cpu_forward", eval_muscles_cpu_forward, "eval_muscles_cpu_forward");
m.def("eval_muscles_cpu_backward", eval_muscles_cpu_backward, "eval_muscles_cpu_backward");
m.def("eval_rigid_fk_cpu_forward", eval_rigid_fk_cpu_forward, "eval_rigid_fk_cpu_forward");
m.def("eval_rigid_fk_cpu_backward", eval_rigid_fk_cpu_backward, "eval_rigid_fk_cpu_backward");
m.def("eval_rigid_id_cpu_forward", eval_rigid_id_cpu_forward, "eval_rigid_id_cpu_forward");
m.def("eval_rigid_id_cpu_backward", eval_rigid_id_cpu_backward, "eval_rigid_id_cpu_backward");
m.def("eval_rigid_tau_cpu_forward", eval_rigid_tau_cpu_forward, "eval_rigid_tau_cpu_forward");
m.def("eval_rigid_tau_cpu_backward", eval_rigid_tau_cpu_backward, "eval_rigid_tau_cpu_backward");
m.def("eval_rigid_jacobian_cpu_forward", eval_rigid_jacobian_cpu_forward, "eval_rigid_jacobian_cpu_forward");
m.def("eval_rigid_jacobian_cpu_backward", eval_rigid_jacobian_cpu_backward, "eval_rigid_jacobian_cpu_backward");
m.def("eval_rigid_mass_cpu_forward", eval_rigid_mass_cpu_forward, "eval_rigid_mass_cpu_forward");
m.def("eval_rigid_mass_cpu_backward", eval_rigid_mass_cpu_backward, "eval_rigid_mass_cpu_backward");
m.def("eval_dense_gemm_cpu_forward", eval_dense_gemm_cpu_forward, "eval_dense_gemm_cpu_forward");
m.def("eval_dense_gemm_cpu_backward", eval_dense_gemm_cpu_backward, "eval_dense_gemm_cpu_backward");
m.def("eval_dense_gemm_batched_cpu_forward", eval_dense_gemm_batched_cpu_forward, "eval_dense_gemm_batched_cpu_forward");
m.def("eval_dense_gemm_batched_cpu_backward", eval_dense_gemm_batched_cpu_backward, "eval_dense_gemm_batched_cpu_backward");
m.def("eval_dense_cholesky_cpu_forward", eval_dense_cholesky_cpu_forward, "eval_dense_cholesky_cpu_forward");
m.def("eval_dense_cholesky_cpu_backward", eval_dense_cholesky_cpu_backward, "eval_dense_cholesky_cpu_backward");
m.def("eval_dense_cholesky_batched_cpu_forward", eval_dense_cholesky_batched_cpu_forward, "eval_dense_cholesky_batched_cpu_forward");
m.def("eval_dense_cholesky_batched_cpu_backward", eval_dense_cholesky_batched_cpu_backward, "eval_dense_cholesky_batched_cpu_backward");
m.def("eval_dense_subs_cpu_forward", eval_dense_subs_cpu_forward, "eval_dense_subs_cpu_forward");
m.def("eval_dense_subs_cpu_backward", eval_dense_subs_cpu_backward, "eval_dense_subs_cpu_backward");
m.def("eval_dense_solve_cpu_forward", eval_dense_solve_cpu_forward, "eval_dense_solve_cpu_forward");
m.def("eval_dense_solve_cpu_backward", eval_dense_solve_cpu_backward, "eval_dense_solve_cpu_backward");
m.def("eval_dense_solve_batched_cpu_forward", eval_dense_solve_batched_cpu_forward, "eval_dense_solve_batched_cpu_forward");
m.def("eval_dense_solve_batched_cpu_backward", eval_dense_solve_batched_cpu_backward, "eval_dense_solve_batched_cpu_backward");
m.def("eval_rigid_integrate_cpu_forward", eval_rigid_integrate_cpu_forward, "eval_rigid_integrate_cpu_forward");
m.def("eval_rigid_integrate_cpu_backward", eval_rigid_integrate_cpu_backward, "eval_rigid_integrate_cpu_backward");
m.def("solve_springs_cpu_forward", solve_springs_cpu_forward, "solve_springs_cpu_forward");
m.def("solve_springs_cpu_backward", solve_springs_cpu_backward, "solve_springs_cpu_backward");
m.def("solve_tetrahedra_cpu_forward", solve_tetrahedra_cpu_forward, "solve_tetrahedra_cpu_forward");
m.def("solve_tetrahedra_cpu_backward", solve_tetrahedra_cpu_backward, "solve_tetrahedra_cpu_backward");
m.def("solve_contacts_cpu_forward", solve_contacts_cpu_forward, "solve_contacts_cpu_forward");
m.def("solve_contacts_cpu_backward", solve_contacts_cpu_backward, "solve_contacts_cpu_backward");
m.def("apply_deltas_cpu_forward", apply_deltas_cpu_forward, "apply_deltas_cpu_forward");
m.def("apply_deltas_cpu_backward", apply_deltas_cpu_backward, "apply_deltas_cpu_backward");
}