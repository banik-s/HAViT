from importlib import import_module

def create_model(img_size, n_classes, args):
    # print(args.model[0:3])

    if args.model == 'vit_original':
        ViT = import_module("models." + args.model).ViT
        patch_size = 4 if img_size == 32 else 8
        model = ViT(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            mlp_dim_ratio=2,
            depth=9,
            heads=12,
            dim_head=192 // 12,
            stochastic_depth=args.sd)

    elif args.model == 'pit':
        PiT = import_module("models." + args.model).PiT
        patch_size = 2 if img_size == 32 else 4
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]

        model = PiT(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=args.channel,
            mlp_dim_ratio=2,
            depth=args.depth,
            heads=args.heads,
            dim_head=dim_head,
            stochastic_depth=args.sd)
            
    elif args.model == 'pit_mod_ver1':
        PiT_mod_ver1 = import_module("models." + args.model).PiT_mod_ver1
        patch_size = 2 if img_size == 32 else 4
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]

        model = PiT_mod_ver1(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=args.channel,
            mlp_dim_ratio=2,
            depth=args.depth,
            heads=args.heads,
            dim_head=dim_head,
            stochastic_depth=args.sd)
            
    elif args.model == 'cait':
        CaiT = import_module("models." + args.model).CaiT
        patch_size = 4 if img_size == 32 else 8
        args.channel = 96  # 1024
        args.heads = 12  # 16
        args.depth = 9  # 12

        model = CaiT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=args.channel,
            depth=args.depth,
            heads=args.heads,
            cls_depth=2,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05)
    elif args.model == 'cait_mod_ver1':
        CaiT_mod_ver1 = import_module("models." + args.model).CaiT_mod_ver1
        patch_size = 4 if img_size == 32 else 8
        args.channel = 96  # 1024
        args.heads = 12  # 16
        args.depth = 9  # 12

        model = CaiT_mod_ver1(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=args.channel,
            depth=args.depth,
            heads=args.heads,
            cls_depth=2,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05)

    elif args.model == 'ScalableViT':
        ScalableViT = import_module("models." + args.model).ScalableViT
        patch_size = 4 if img_size == 32 else 8
        args.channel = 96  # 64
        args.heads = (2, 4, 8, 16)  # 16
        args.depth = (2, 2, 20, 2)  # 12

        model = ScalableViT(
            num_classes=n_classes,
            dim=args.channel,
            depth=args.depth,
            heads=args.heads,
            ssa_dim_key=(
                40,
                40,
                40,
                32),
            reduction_factor=(
                8,
                4,
                2,
                1),
            window_size=(
                2,
                2,
                None,
                None),
            dropout=0.1)

    
            

    elif args.model == 'ViT_lr_modified_sps':
        ViT_lr_modified_sps = import_module("models." + args.model).ViT
        patch_size = 4 if img_size == 32 else 8

        model = ViT_lr_modified_sps(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    
    # Assuming `args` is passed to the main function, and it's accessible at this point.

    elif args.model == 'vit_modified_relu':
	    from models.vit_modified_relu import vit_modified_relu

	    # Determine patch size based on the image size
	    patch_size = 4 if img_size == 32 else 8

	    # Pass `args` to `vit_modified_relu` model constructor if needed
	    model = vit_modified_relu(
		image_size=img_size,
		patch_size=patch_size,
		num_classes=n_classes,
		dim=192,
		depth=9,
		heads=12,
		mlp_dim=512,
		pool='cls',
		channels=3,
		dim_head=64,
		dropout=0.,
		emb_dropout=0.,
		args=args  # Ensure args is passed if vit_modified_relu needs it
	    )

            
    elif args.model == 'vit_modified_quantized':
        ViT_lr_modified_sps = import_module("models." + args.model).ViT
        patch_size = 4 if img_size == 32 else 8

        model = ViT_lr_modified_sps(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    elif args.model == 'vit_modified_hardmax':
        ViT_lr_modified_sps = import_module("models." + args.model).ViT
        patch_size = 4 if img_size == 32 else 8

        model = ViT_lr_modified_sps(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    elif args.model == 'vit_modified_low_rank':
        ViT_lr_modified_sps = import_module("models." + args.model).ViT
        patch_size = 8 if img_size == 32 else 8

        model = ViT_lr_modified_sps(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    elif args.model == 'vit_modified_polynomial':
        ViT_lr_modified_sps = import_module("models." + args.model).ViT
        patch_size = 4 if img_size == 32 else 8

        model = ViT_lr_modified_sps(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    elif args.model == 'vit_modified_kernel':
        ViT_lr_modified_sps = import_module("models." + args.model).ViT
        patch_size = 4 if img_size == 32 else 8

        model = ViT_lr_modified_sps(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    elif args.model == 'ViT_lr_modified_SimA':
        ViT_lr_modified_sps = import_module("models." + args.model).ViT
        patch_size = 4 if img_size == 32 else 8

        model = ViT_lr_modified_sps(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
    
    elif args.model == 'vitlucidrains_mod_ver1':
        vitlucidrains_mod_ver1 = import_module("models." + args.model).vitlucidrains_mod_ver1
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver1(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
    elif args.model == 'vitlucidrains_mod_ver2':
        vitlucidrains_mod_ver2 = import_module("models." + args.model).vitlucidrains_mod_ver2
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver2(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
	
    elif args.model == 'vitlucidrains_mod_ver3':
        vitlucidrains_mod_ver3 = import_module("models." + args.model).vitlucidrains_mod_ver3
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver3(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
	
	
	
	
    elif args.model == 'vitlucidrains_mod_ver4':
        vitlucidrains_mod_ver4 = import_module("models." + args.model).vitlucidrains_mod_ver4
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver4(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    elif args.model == 'vitlucidrains_mod_ver5':
        vitlucidrains_mod_ver5 = import_module("models." + args.model).vitlucidrains_mod_ver5
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver5(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
            
    elif args.model == 'vitlucidrains_mod_ver6':
        vitlucidrains_mod_ver6 = import_module("models." + args.model).vitlucidrains_mod_ver6
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver6(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)


    elif args.model == 'vitlucidrains_mod_ver7':
        vitlucidrains_mod_ver7 = import_module("models." + args.model).vitlucidrains_mod_ver7
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver7(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)


    elif args.model == 'vitlucidrains_mod_ver8':
        vitlucidrains_mod_ver8 = import_module("models." + args.model).vitlucidrains_mod_ver8
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver8(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)

    elif args.model == 'vitlucidrains_mod_ver9':
        vitlucidrains_mod_ver9 = import_module("models." + args.model).vitlucidrains_mod_ver9
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver9(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
            
    elif args.model == 'vitlucidrains_mod_ver10':
        vitlucidrains_mod_ver10 = import_module("models." + args.model).vitlucidrains_mod_ver10
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver10(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
    
    elif args.model == 'vitlucidrains_mod_ver11':
        vitlucidrains_mod_ver11 = import_module("models." + args.model).vitlucidrains_mod_ver11
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver11(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
            
    elif args.model == 'vitlucidrains_mod_ver12':
        vitlucidrains_mod_ver12 = import_module("models." + args.model).vitlucidrains_mod_ver12
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver12(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
    elif args.model == 'vitlucidrains_mod_ver13':
        vitlucidrains_mod_ver13 = import_module("models." + args.model).vitlucidrains_mod_ver13
        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains_mod_ver13(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
            
            
    
            
    # Ensure the final model definition and return is outside all conditionals
    elif args.model == 'ViT_Hybrid':
        from models.ViT_Hybrid import ViT_Hybrid

        patch_size = 4 if img_size == 32 else 8

        model = ViT_Hybrid(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
            
            
    elif args.model == 'vitlucidrains':
        from models.vitlucidrains import vitlucidrains

        patch_size = 4 if img_size == 32 else 8

        model = vitlucidrains(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
            
            
    # Ensure the final model definition and return is outside all conditionals
    elif args.model == 'ViT_Hybrid_01':
        from models.ViT_Hybrid_01 import ViT_Hybrid_01

        patch_size = 4 if img_size == 32 else 8

        model = ViT_Hybrid_01(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
    
    elif args.model == 'ViT_Hybrid_02':
        from models.ViT_Hybrid_02 import ViT_Hybrid_02

        patch_size = 4 if img_size == 32 else 8

        model = ViT_Hybrid_02(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
          
            
            
    elif args.model == 'ViT_ReLU_Sparsemax':
        from models.ViT_ReLU_Sparsemax import ViT_ReLU_Sparsemax

        patch_size = 4 if img_size == 32 else 8

        model = ViT_ReLU_Sparsemax(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)
    elif args.model == 'ViT_Dynamic':
        from models.ViT_Dynamic import ViT_Dynamic

        patch_size = 4 if img_size == 32 else 8

        model = ViT_Dynamic(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.)


    return model
