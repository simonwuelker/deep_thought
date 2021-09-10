use proc_macro::{TokenStream, TokenTree};
use quote::quote;
use syn::{
    parse::{Error, Parse, ParseStream, Result},
    parse_macro_input,
    spanned::Spanned,
    DeriveInput, Expr,
    Expr::{Call, MethodCall},
    Ident, Lit, Token, Type, Visibility,
};

struct Network {
    visibility: Visibility,
    name: Ident,
    ty: Type,
    init: Expr,
}

impl Parse for Network {
    fn parse(input: ParseStream) -> Result<Self> {
        let visibility: Visibility = input.parse()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty: Type = input.parse()?;
        input.parse::<Token![=]>()?;
        let init: Expr = input.parse()?;
        input.parse::<Token![;]>()?;
        Ok(Network {
            visibility,
            name,
            ty,
            init,
        })
    }
}

/// Try parsing the provided argument into a integer literal and throw a compiler error if the conversion fails
macro_rules! int_lit_from_fn_arg {
    ($arg:expr) => {
        if let Expr::Lit(first_expr_lit) = &$arg {
            if let Lit::Int(int_lit) = &first_expr_lit.lit {
                match int_lit.base10_parse::<usize>() {
                    Ok(int) => int,
                    Err(error) => return error.into_compile_error().into(),
                }
            } else {
                return Error::new($arg.span(), "argument is not an integer literal")
                    .into_compile_error()
                    .into();
            }
        } else {
            return Error::new($arg.span(), "argument is not a literal")
                .into_compile_error()
                .into();
        }
    };
}

// this macro will very likely break in the future when deep_thought's syntax changes
#[proc_macro]
pub fn neural_network(input: TokenStream) -> TokenStream {
    // Parse the TokenStream into an Abstract Syntax Tree (AST)
    let cloned_inp = input.clone();
    let Network {
        visibility,
        name,
        ty,
        init,
    } = parse_macro_input!(cloned_inp as Network);

    let mut num_parameters = 0;
    let mut fn_ref = &init;
    while let MethodCall(expr_method_call) = fn_ref {
        if expr_method_call.method == "add_layer" {
            if let Some(x) = expr_method_call.args.first() {
                let mut some_ref = x;
                // ignore all layer.activation method calls
                while let MethodCall(inner_expr_method_call) = some_ref {
                    some_ref = &inner_expr_method_call.receiver;
                }
                if let Call(inner_expr_call) = some_ref {
                    let first_arg = inner_expr_call.args.first().unwrap();
                    let second_arg = inner_expr_call.args.last().unwrap();

                    let in_size = int_lit_from_fn_arg!(first_arg);
                    let out_size = int_lit_from_fn_arg!(second_arg);

                    num_parameters += (in_size + 1) * out_size;
                }
            }
        }

        fn_ref = &expr_method_call.receiver;
    }

    let constant: TokenStream = quote!{
        const _NUM_PARAMETERS = #num_parameters;
    }.into();
    input.clone().extend(constant);
    input
}
